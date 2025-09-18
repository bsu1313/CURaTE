import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerBase, PreTrainedModel
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple, Any, Dict
import os, sys, json, math, random, argparse, tqdm, re
from rouge_score import rouge_scorer
import numpy as np
from datasets import load_dataset
import math
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
import time


rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def abspath(*p):
    return os.path.abspath(os.path.join(*p))

def _mean(x: List[float]): return float(np.mean(x)) if x else 0.0

class QADataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def mapped_question(origin_id: int, id2question, ID_MAP) -> List[str]:
    # print("origin_id: ", origin_id)
    # print("ID_MAP[str(origin_id)]: ", ID_MAP[str(origin_id)])
    try:
        mapped_ids = ID_MAP[str(origin_id)][f"forget_data_top3_ids"]
        # return_values = [id2question[mid] for mid in mapped_ids if mid in id2question]
        # print("len of return_values: ", len(return_values))
        # for r in return_values:
        #     print("mapped question: ", r)
        return [id2question[mid] for mid in mapped_ids if mid in id2question]
    except (KeyError, IndexError):
        return id2question[origin_id]

def wrap_prompt(p, if_llama):
    if 'llama-3' in if_llama or 'llama_3' in if_llama:
        question_start_token = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Jul 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        question_end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'llama-2' in if_llama or 'llama_2' in if_llama:
        question_start_token = "<s>[INST] "
        question_end_token = " [/INST]"
    else:
        raise ValueError('Please provide llama model')
    # print("wrapped prompt: ", f"{question_start_token}{p}{question_end_token}")
    return f"{question_start_token}{p}{question_end_token}"

def batched_generate(model, tok, prompts, gen_length):
    # print("prompts: ", prompts)
    inputs = tok(prompts, return_tensors="pt",
                 padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        if gen_length is None:
            outs = model.generate(**inputs,
                                max_length = 256,
                                do_sample=False,
                                eos_token_id=tok.eos_token_id,
                                use_cache=False)
        else:
            outs = model.generate(**inputs,
                    max_new_tokens=gen_length,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    use_cache=False)

    results = []
    for prompt, generated_ids in zip(prompts, outs):
        # Decode the full output without skipping special tokens
        full_text = tok.decode(
            generated_ids,
            skip_special_tokens=True,
            # skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        ).strip()

        # Also decode the prompt the same way
        prompt_text = tok.decode(
            tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        # Remove the prompt text from the start
        if full_text.startswith(prompt_text):
            answer = full_text[len(prompt_text):].strip()
        else:
            # fallback: search for prompt text inside output
            idx = full_text.find(prompt_text)
            if idx != -1:
                answer = full_text[idx + len(prompt_text):].strip()
            else:
                # fallback: just return the full text
                answer = full_text
        results.append(answer)
    # print("results: ", results)
    return results

def build_WD_prompt(SENTENCE: str, OPT1, OPT2) -> str:
    user_msg = (
        f"""Choose the option that best fills "_" in the sentence.
Return ONLY the chosen option EXACTLY as written (same case and spacing). Output nothing else.
Important: Do NOT repeat the sentence in your answer.

Sentence: {SENTENCE}
Options:
- {OPT1}
- {OPT2}

The correct option is:
"""
    )
    return user_msg

# =========================
# MLP Classifier
# =========================
HIDDEN_DIM = 512
class MLPBin(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1)  # logits
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

@torch.no_grad()
def extract_batch_penultimate_embeddings(
    queries: list[str],
    tokenizer,
    model,
    max_len: int = 512,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Returns (N, H) NumPy array: average of penultimate hidden states
    for each query string.
    """
    model.eval()

    # Choose device from model params
    first_param = next(iter(model.parameters()))
    device = first_param.device

    feats = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,   # pad within batch
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True)
        penultimate = out.hidden_states[-2]              # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)       # [B, T, 1]
        penultimate = penultimate * mask

        lengths = mask.sum(dim=1).clamp(min=1)           # [B, 1]
        avg_emb = penultimate.sum(dim=1) / lengths       # [B, H]

        feats.append(avg_emb.cpu().float().numpy())

    return np.concatenate(feats, axis=0)  # [N, H]


@dataclass
class BeamItem:
    input_ids: torch.LongTensor
    gen_only_ids: List[int]
    contaminated: bool
    last_cost: float
    is_finished: bool


@torch.no_grad()
def generate_with_beam_penalty_semantic(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    answer: str,
    k: int = 7,
    *,
    # new args for semantic penalty
    embedder: Any,                         # e.g., SentenceTransformer
    alpha: float = 1.0,                    # scales cosine similarity when below threshold
    sim_threshold: float = 0.5,            # >= threshold -> ∞ penalty
    embedder_device: Optional[str] = None, # e.g., "cuda" or "cpu"; if None, use embedder default
    #
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    per_beam_topk: Optional[int] = None,
    forbid_overlap_scope: str = "generated_only",
) -> Tuple[str, List[BeamItem]]:
    """
    Beam search with:
      cost(step) = -log p(last_token|prefix) + (∞ if token overlaps 'answer') + semantic_penalty

    semantic_penalty:
      last_word = last whitespace-delimited token in decoded generated text
      sim = cosine( embed(last_word), embed(answer) )
      penalty = ∞ if sim >= sim_threshold else alpha * sim
    """
    if device is None:
        device = next(model.parameters()).device
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if per_beam_topk is None:
        per_beam_topk = max(50, 5 * k)

    model.eval()

    # Tokenize prompt and build forbidden set from answer tokens
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    special_ids = set(t for t in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id] if t is not None)
    forbidden = set([t for t in answer_ids if t not in special_ids])

    # --- Embedding helpers & caches ---
    # cache: text -> L2-normalized torch vector (on CPU)
    _emb_cache: Dict[str, torch.Tensor] = {}

    def _to_unit(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(p=2) + 1e-12)

    def _encode_text(text: str) -> torch.Tensor:
        if text in _emb_cache:
            return _emb_cache[text]
        # Sentence-Transformers .encode may return numpy or torch; request tensor if available
        try:
            vec = embedder.encode(text, convert_to_tensor=True, device=embedder_device)
            if not isinstance(vec, torch.Tensor):
                vec = torch.as_tensor(vec)
        except TypeError:
            # Fallback for ST versions without device kwarg
            vec = embedder.encode(text, convert_to_tensor=True)
            if not isinstance(vec, torch.Tensor):
                vec = torch.as_tensor(vec)
        vec = vec.detach().cpu().float()
        vec = _to_unit(vec)
        _emb_cache[text] = vec
        return vec

    answer_emb = _encode_text(answer)

    def _last_word(decoded_text: str) -> str:
        # take last whitespace-delimited token; strip common punctuation
        stripped = decoded_text.rstrip()
        if not stripped:
            return ""
        word = stripped.split()[-1]
        return word.strip(".,;:!?\"'()[]{}<>")

    # Init beams
    beams: List[BeamItem] = [
        BeamItem(input_ids=prompt_ids.clone(), gen_only_ids=[], contaminated=False, last_cost=0.0, is_finished=False)
        for _ in range(k)
    ]

    def overlap_penalty(candidate_gen_ids: List[int], new_token_id: int, contaminated: bool) -> float:
        if contaminated:
            return float("inf")
        if forbid_overlap_scope == "generated_only":
            return float("inf") if new_token_id in forbidden else 0.0
        elif forbid_overlap_scope == "full_candidate":
            return float("inf") if (new_token_id in forbidden or any(t in forbidden for t in candidate_gen_ids)) else 0.0
        else:
            raise ValueError("forbid_overlap_scope must be 'generated_only' or 'full_candidate'")

    def semantic_penalty(new_gen_ids: List[int]) -> float:
        # Decode generated portion to get last word
        if not new_gen_ids:
            return 0.0
        decoded = tokenizer.decode(new_gen_ids, skip_special_tokens=True)
        lw = _last_word(decoded)
        if not lw:
            return 0.0
        lw_emb = _encode_text(lw)
        # cosine sim on CPU unit vectors
        sim = float((lw_emb * answer_emb).sum().item())
        if sim >= sim_threshold:
            return float("inf")
        return alpha * sim

    # Generation loop
    for _step in range(max_new_tokens):
        active = [b for b in beams if not b.is_finished]
        if not active:
            break

        max_len = max(b.input_ids.shape[1] for b in active)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        batch = []
        for b in active:
            pad_len = max_len - b.input_ids.shape[1]
            if pad_len > 0:
                padded = torch.cat(
                    [b.input_ids, torch.full((1, pad_len), pad_id, dtype=b.input_ids.dtype, device=device)],
                    dim=1
                )
            else:
                padded = b.input_ids
            batch.append(padded)
        batch_input = torch.cat(batch, dim=0)

        out = model(batch_input)
        logits = out.logits
        next_logits = logits[:, -1, :]
        logprobs = F.log_softmax(next_logits, dim=-1)

        candidate_pool: List[BeamItem] = []
        for i, b in enumerate(active):
            top_logp, top_ids = torch.topk(logprobs[i], per_beam_topk, dim=-1)
            for j in range(per_beam_topk):
                tok_id = int(top_ids[j].item())
                lp = float(top_logp[j].item())
                is_eos = (tok_id == eos_token_id)

                # Overlap penalty (token-level)
                pen_overlap = overlap_penalty(b.gen_only_ids, tok_id, b.contaminated)

                # Build new candidate sequences (needed for semantic penalty)
                new_input = torch.cat(
                    [b.input_ids, torch.tensor([[tok_id]], device=device, dtype=b.input_ids.dtype)], dim=1
                )
                new_gen = b.gen_only_ids + [tok_id]
                contaminated_next = b.contaminated or (pen_overlap == float("inf"))

                # Semantic penalty (answer vs last word)
                pen_sem = 0.0 if contaminated_next else semantic_penalty(new_gen)

                # Final step cost
                step_cost = -lp + pen_overlap + pen_sem

                candidate_pool.append(
                    BeamItem(
                        input_ids=new_input,
                        gen_only_ids=new_gen,
                        contaminated=contaminated_next or math.isinf(pen_sem),
                        last_cost=step_cost,
                        is_finished=is_eos,
                    )
                )

        candidate_pool.sort(key=lambda x: (math.inf if math.isinf(x.last_cost) else x.last_cost))
        beams = candidate_pool[:k]

        if all(b.is_finished for b in beams):
            break

    finished = [b for b in beams if b.is_finished and not math.isinf(b.last_cost)]
    pick_from = finished if finished else [b for b in beams if not math.isinf(b.last_cost)] or beams
    pick_from.sort(key=lambda x: x.last_cost if not math.isinf(x.last_cost) else math.inf)
    best = pick_from[0]

    text = tokenizer.decode(best.gen_only_ids, skip_special_tokens=True)
    return text, beams



def eval_subset(model, tok, clf, sent_model, model_name, name, ds, gen_length, id2question, ID_MAP, device, batch_size=4):

    def identity_collate(batch):
        return batch

    # print("len of ds: ", len(ds)) # forget01: 40
    dl = DataLoader(QADataset(ds), batch_size=batch_size, collate_fn=identity_collate)

    metrics = {k:[] for k in
               ("truth_ratio","truth_prob","rougeL","acc")}
    samples = []

    total_positives = 0
    total_negatives = 0
    for batch in tqdm.tqdm(dl, desc=f"Eval {name}"):
        prompts_1, questions_1, correct_1, incorrect_1, preds_1, ans_1, beam_times = [], [], [], [], [], [], []
        for item in batch:
            # print("item: ", item)
            question = item["paraphrased_question"] if name == "forget" else item["question"]
            questions_1.append(question)
            ref_q = mapped_question(item["id"], id2question, ID_MAP)
            ans_1.append(ref_q[0])

            prompts_1.append(wrap_prompt(question, model_name.lower()))
            correct_1.append(item["answer"])
            if name == "winogrande":
                incorrect_1.append(item["incorrect_answer"])
            # print("prompts_1: ", prompts_1)
            # print("correct_1: ", correct_1)

        avg_embs = extract_batch_penultimate_embeddings(questions_1, tok, model, max_len=512, batch_size=batch_size,)  # (B, H)
        avg_embs = torch.from_numpy(avg_embs)
        if avg_embs.ndim == 1:          # (H,) -> (1, H)
            avg_embs = avg_embs.unsqueeze(0)
        avg_embs = avg_embs.to(device)   # (B, H)
        logits = clf(avg_embs)              # shape (B, 1)
        probs = torch.sigmoid(logits).squeeze(-1)
        preds_1 = (probs >= 0.5).long().tolist()
        if not isinstance(preds_1, list):
            preds_1 = [preds_1]
        # print("questions_1: ", questions_1)
        # print("ans_1: ", ans_1)
        # print("avg embs shape: ", avg_embs.shape)
        # print("probs: ", probs)
        # print("preds_1: ", preds_1)
        total_positives += sum(preds_1)
        total_negatives += len(preds_1) - sum(preds_1)

    agg = {}
    agg[f"{name} positives"] = total_positives
    agg[f"{name} negatives"] = total_negatives
    # agg["total num samples"] = len(ds)
    return agg, samples



def main():

    model_size = "7B" # 1B, 7B
    task = "RETURN" # TOFU, TruthfulQA, ScienceQA, RETURN
    stage = 1
    if stage == 1:
        split = "1"
    elif stage == 2:
        split = "12"
    elif stage == 3:
        split = "123"
    
    # Configuration
    if task == "TOFU":
        if model_size == "1B":
            model_path = "models/tofu_Llama-3.2-1B-Instruct_full"
        elif model_size == "7B":
            model_path = "models/tofu_Llama-2-7b-chat-hf_full"
        else:
            raise ValueError(f"Unknown model size: {model_size}")
    else:
        if model_size == "1B":
            model_path = "models/Llama-3.2-1B-Instruct"
        elif model_size == "7B":
            model_path = "models/Llama-2-7b-chat-hf"
        else:
            raise ValueError(f"Unknown model size: {model_size}")

    device_map = "auto"
    batch_size = 4
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
    model = model.eval()

    device = torch.device("cuda")
    hidden_size = model.config.hidden_size
    print("Hidden size:", hidden_size)
    in_dim = hidden_size
    clf = MLPBin(in_dim, HIDDEN_DIM).to(device)
    prev_save_dir = "models/guard_" + task + f"_{model_size}_stage{stage}"
    prev_ckpt = os.path.join(prev_save_dir, "mlp_best.pt")
    ckpt = torch.load(prev_ckpt, map_location=device)
    clf.load_state_dict(ckpt["model"])
    clf.eval()

    model_dir = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    sent_model = SentenceTransformer(model_dir)


    # # sample_question = "What does Hsiao Yun-Hwa identify as in terms of gender?"
    # sample_question = "What gender is author Basil Mahfouz Al-Kuwaiti?"
    # inputs = tok(sample_question, return_tensors="pt")
    # embed_device = model.get_input_embeddings().weight.device
    # inputs = {k: v.to(embed_device) for k, v in inputs.items()}
    # generated_answer = model.generate(**inputs)
    # print("generated answer: ", tok.decode(generated_answer[0], skip_special_tokens=False))

    
    gen_length = None
    if task == "TOFU":
        split_dir = "TOFU_NEW/"
        splits = {}
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}.json"), encoding="utf-8") as f:
            splits["forget"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"retain_perturbed.json"), encoding="utf-8") as f:
            splits["retain"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}_NU.json"), encoding="utf-8") as f:
            splits["forget_NU"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"real_authors.json"), encoding="utf-8") as f:
            splits["real_authors"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"world_facts.json"), encoding="utf-8") as f:
            splits["world_facts"] = json.load(f)
        
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}.json"), encoding="utf-8") as f:
            forget_split = json.load(f)
            id2question: dict[int, str] = {ex["id"]: ex["answer"] for ex in forget_split}
        MAPPING_PATH = Path(split_dir) / f"stage{split[-1]}" / f"TOFU_to_forget{split}_top3_guard.json"
        with MAPPING_PATH.open("r", encoding="utf-8") as f:
            ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)
    elif task == "TruthfulQA":
        input_file = "truthfulQA/truthfulQA_continual_setting/truthfulQA_all_augmented_ID.json"
        split_file = "truthfulQA/truthfulQA_continual_setting/TruthfulQA_split_ids.json"
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)
        with open(split_file, encoding="utf-8") as f:
            split_ids = json.load(f)
        
        stage1_ids = set(split_ids["stage1"])
        stage1_stage2_ids = set(split_ids["stage1"]) | set(split_ids["stage2"])
        stage1_stage2_stage3_ids = (set(split_ids["stage1"]) | set(split_ids["stage2"]) | set(split_ids["stage3"]))
        if stage == 1:
            combined_ids = stage1_ids
        elif stage == 2:
            combined_ids = stage1_stage2_ids
        elif stage == 3:
            combined_ids = stage1_stage2_stage3_ids
        # filtered_data = [example for example in data if example["id"] in combined_ids]

        splits = {}
        splits["forget"] = [
            {
                "paraphrased_question": example["paraphrased_question"],
                "answer": [s.strip() for s in example["Incorrect Answers"].split(";")]
            }
            for example in data if example["id"] in combined_ids]
        splits["contrastive"] = [
            {
                "question": example["contrastive_question"],
                "answer": example["contrastive_answer"]
            }
            for example in data if example["id"] in combined_ids]
        ds = load_dataset("tau/commonsense_qa", split="validation")
        splits["commonsense"] = []
        for ex in ds:
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            gold_text = dict(zip(labels, texts))[ex["answerKey"]]
            choices = list(zip(labels, texts))
            choice_block = "\n".join([f"{label}. {text}" for label, text in choices])
            usr_msg = (
                f"{ex['question']}\n\nChoices:\n{choice_block}\n\n"
                "Include both the letter and the full correct answer."
            )
            item = {
                "question": usr_msg,
                "answer": gold_text
            }
            splits["commonsense"].append(item)
    elif task == "RETURN":
        splits = {}
        if model_size == "1B":
            split_dir = "RETURN_NEW_DATASET/Meta-Llama-3.2-1B-Instruct_dataset/"
        elif model_size == "7B":
            split_dir = "RETURN_NEW_DATASET/Meta-Llama-2-7B-chat_dataset/"
        with open(os.path.join(split_dir, f"stage_{stage-1}_forget_paraphrased.json"), encoding="utf-8") as f:
                splits["forget"] = json.load(f)
                for item in splits["forget"]:
                    item["paraphrased_question"] = item["paraphrased_instruction"]
                    item["answer"] = item["gold_answer"]
        with open(os.path.join(split_dir, f"stage_{stage-1}_retain_used.json"), encoding="utf-8") as f:
                splits["retain_used"] = json.load(f)
                for item in splits["retain_used"]:
                    item["answer"] = item["gold_answer"]
        with open(os.path.join(split_dir, f"stage_{stage-1}_retain_not_used.json"), encoding="utf-8") as f:
                splits["retain_not_used"] = json.load(f)
                for item in splits["retain_not_used"]:
                    item["answer"] = item["gold_answer"]
        with open(os.path.join(split_dir, f"non_target.json"), encoding="utf-8") as f:
                splits["non_target"] = json.load(f)
                for item in splits["non_target"]:
                    item["answer"] = item["gold_answer"]
        with open(os.path.join(split_dir, f"stage_{stage-1}_near_utility.json"), encoding="utf-8") as f:
                splits["near_utility"] = json.load(f)
                for item in splits["near_utility"]:
                    item["question"] = item["contrastive_instruction"]
                    item["answer"] = item["contrastive_answer"]
        with open(os.path.join(split_dir, f"winogrande_xs_validation.json"), encoding="utf-8") as f:
                splits["winogrande"] = json.load(f)
                for item in splits["winogrande"]:
                    item["question"] = build_WD_prompt(
                        item["sentence"], item["option1"], item["option2"]
                    )
                    if item["answer"] == "1":
                        item["answer"] = item["option1"]
                        item["incorrect_answer"] = item["option2"]
                    else:
                        item["answer"] = item["option2"]
                        item["incorrect_answer"] = item["option1"]
        
        with open(os.path.join(split_dir, f"stage_{stage-1}_forget.json"), encoding="utf-8") as f:
            forget_split = json.load(f)
            id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in forget_split}

        MAPPING_PATH = Path(split_dir) / f"RETURN_stage_{stage-1}_top3_guard.json"
        with MAPPING_PATH.open("r", encoding="utf-8") as f:
            ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)

    # for name, ds in splits.items():
    #     print("name: ", name)
    #     print("sample: ", ds[0])
    #     print("len: ", len(ds))
    

    result: Dict[str,Dict] = {}
    for name, ds in splits.items():
        agg, detail = eval_subset(model, tok, clf, sent_model, model_path, name, ds,
                                  gen_length, id2question, ID_MAP, device, batch_size=batch_size)
        result[name] = {"metrics": agg, "samples": detail}
        print(f"[{name}] {json.dumps(agg, indent=2, ensure_ascii=False)}")
    final_metrics = {name: res["metrics"] for name, res in result.items()}
    print("\n==== Final Aggregated Metrics ====")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print("Finished eval_tofu stage ", stage, " for task ", task, " for model size ", model_size)

if __name__ == "__main__":
    main()
