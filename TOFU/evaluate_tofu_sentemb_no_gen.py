#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys, json, math, random, argparse, tqdm, re
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
import deepspeed
import transformers
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer


from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import random

REFUSAL_PATH = Path("./refusal_answer.json")  
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))

def get_available_cache_dir():
    preferred = Path("/home/.cache")
    fallback = Path("/home/plowcow/.cache")

    if preferred.exists() and os.access(preferred, os.W_OK):
        return str(preferred)
    else:
        return str(fallback)


map_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

def build_forget_index(forget_ds, bs=128):

    f_texts = forget_ds["question"]
    f_embs  = map_model.encode(
        f_texts, batch_size=bs, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=True)
    return f_texts, f_embs        # list[str], Tensor[N,768]

def match_forget_questions(raw_qs, f_texts, f_embs,
                           mapping: Dict[str,str]) -> List[str]:

    # raw_q_sample = raw_qs[:5]
    # sample_mapping = [mapping[q] for q in raw_q_sample if q in mapping]
    # print("Sample raw_qs:", raw_q_sample)
    # print("Sample mapping:", sample_mapping)
    # first_five_keys = list(mapping.keys())[:5]
    # for key in first_five_keys:
    #     print(f"Mapping for '{key}': {mapping[key]}")

    need_compute = [q for q in raw_qs if q not in mapping]
    if need_compute:
        rq_embs = map_model.encode(
            need_compute, batch_size=len(need_compute),
            convert_to_tensor=True, normalize_embeddings=True)

        sim = rq_embs @ f_embs.T                # [M, N]
        best_idx = sim.argmax(dim=1).tolist()
        for q, idx in zip(need_compute, best_idx):
            mapping[q] = f_texts[idx]          

    return [mapping[q] for q in raw_qs]

def mapped_question(origin_id: int, id2question, ID_MAP) -> List[str]:

    # print("origin_id: ", origin_id)
    # print("ID_MAP[str(origin_id)]: ", ID_MAP[str(origin_id)])
    try:
        mapped_ids = ID_MAP[str(origin_id)][f"forget_data_top3_ids"]
        return [id2question[mid] for mid in mapped_ids if mid in id2question]
    except (KeyError, IndexError):
        return id2question[origin_id]

def mapped_cossim(origin_id: int, ID_MAP) -> List[str]:
    mapped_ids = ID_MAP[str(origin_id)][f"forget_data_top3_cossim"]
    return mapped_ids

class QADataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def wrap_prompt(p, if_llama):
    if 'llama-3' in if_llama or 'llama_3' in if_llama:
        question_start_token = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Jul 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        question_end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'llama-2' in if_llama or 'llama_2' in if_llama:
        question_start_token = "[INST] "
        question_end_token = " [/INST]"
    else:
        raise ValueError('Please provide llama model')

    return f"{question_start_token}{p}{question_end_token}"
def build_llama2_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,   # Return plain text prompt, not token IDs
        add_generation_prompt=True  # Adds the assistant's turn prefix
    )
    return prompt


rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
# st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def _mean(x: List[float]): return float(np.mean(x)) if x else 0.0

def truth_ratio(tp: float, fp: List[float]):
    return (np.mean(fp)+1e-12)/(tp+1e-12)

def acc_contains(pred, truth):
    return int(bool(re.search(re.escape(truth), pred, re.I)))



# def load_model(base, lora, ds_cfg, cache_dir, dtype=torch.float16):
def load_model(base, ds_cfg, cache_dir, dtype=torch.float16):
    cfg = transformers.AutoConfig.from_pretrained(base)
    cfg.tp_size = 1          # disable HF-TP

    model = AutoModelForCausalLM.from_pretrained(
        base, config=cfg, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map=None,
        cache_dir=cache_dir
    )
    # model = PeftModel.from_pretrained(model, lora).merge_and_unload()

    engine = deepspeed.init_inference(
        model, dtype=dtype, kernel_inject=False,
        replace_method="none", config=json.load(open(ds_cfg))
    )
    if "Llama-2-7b" in base:
        tok_name = "meta-llama/Llama-2-7b-chat-hf"
    elif "Llama-3.2-1B-Instruct" in base:
        tok_name = "meta-llama/Llama-3.2-1B-Instruct"
    else:
        tok_name = base
    tok = AutoTokenizer.from_pretrained(
        tok_name,
        use_fast=False,
        padding_side="left",
        cache_dir= get_available_cache_dir(),
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    
    
    return engine.module, tok





def batched_generate(model, tok, prompts):
    # print("prompts: ", prompts)
    inputs = tok(prompts, return_tensors="pt",
                 padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        outs = model.generate(**inputs,
                              # max_new_tokens=256,
                              max_length = 256,
                              do_sample=False,
                              # min_new_tokens=4,
                              eos_token_id=tok.eos_token_id,
                              use_cache=False)

    results = []
    for prompt, generated_ids in zip(prompts, outs):
        # Decode the full output without skipping special tokens
        full_text = tok.decode(
            generated_ids,
            skip_special_tokens=True,
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
    return results


def seq_prob(model, tok, text):
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids, use_cache=False)
    logit = out.logits[:,:-1]
    label = ids.input_ids[:,1:]
    nll = torch.nn.functional.cross_entropy(
        logit.transpose(-1,-2), label,
        ignore_index=tok.pad_token_id, reduction="sum")
    avg = nll / (label != tok.pad_token_id).sum()
    return math.exp(-avg.item())


def eval_subset(name, ds, id2question, ID_MAP, batch_size=4):
    # dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    def identity_collate(batch):
        return batch

    # print("len of ds: ", len(ds)) # forget01: 40
    dl = DataLoader(QADataset(ds), batch_size=batch_size, collate_fn=identity_collate)

    metrics = {k:[] for k in
               ("truth_ratio","truth_prob","rougeL","acc")}
    samples = []
    par_positives = 0
    par_negatives = 0

    for batch in tqdm.tqdm(dl, desc=f"Eval {name}"):
        prompts_1, questions_1, preds_1, correct_1 = [], [], [], []
        for item in batch:
            # print("item: ", item)
            # question = item["paraphrased_question"] if "paraphrased_question" in item.keys() else item["question"]
            question = item["paraphrased_question"] if name == "forget" else item["question"]
            # print("question: ", question)
            questions_1.append(question)
            ref_q = mapped_question(item["id"], id2question, ID_MAP)
            cos_sim = mapped_cossim(item["id"], ID_MAP)
            # print("cos_sim: ", cos_sim)
            max_cos_sim = max(float(x) for x in cos_sim) if cos_sim else 0.0

            if max_cos_sim > 0.8: # 0.8
                match = True
            else:
                match = False
            # match = False

            # print("match: ", match)
            if not match:
                preds_1.append(0)
                par_negatives += 1
            else:
                preds_1.append(1)
                par_positives += 1

            correct_1.append(item["answer"])
            # print("prompts_1: ", prompts_1)
            # print("correct_1: ", correct_1)

    agg = {k: _mean(v) for k, v in metrics.items()}
    agg["positives"] = par_positives
    agg["negatives"] = par_negatives
    return agg


def load_split(name, cache):
    return load_dataset("locuslab/TOFU", name,
                        cache_dir=cache, split="train")


def get_seen_unseen(ds, ratio=0.8, seed=1000):
    random.seed(seed)
    idx_seen = random.sample(range(len(ds)), int(len(ds)*ratio))
    idx_unseen = list(set(range(len(ds))) - set(idx_seen))
    return ds.select(idx_seen), ds.select(idx_unseen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_config", default="ds_config.json")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", default="./eval_results")
    ap.add_argument("--cache_dir", default=get_available_cache_dir())
    ap.add_argument("--local_rank", type=int, default=-1, help="(set by deepspeed)")
    ap.add_argument("--split_dir", default="TOFU_NEW")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ablation = 1  # 0, 1, 2, 3, 4, 5, 6
    split = "123"  # "1", "12", "123"

    ablation_files = [
        "NQ_CURE_12K_a",
        "NQ_CURE_18K_a",
        "NQ_CURE_18K_a_no_b",
        "NQ_CURE_NO_HN_18K_a",
        "NQ_CURE_NO_HN_18K_a_no_b",
        "TQ_CURE_18K_a",
        "no_finetuning"
    ]

    splits = {}
    with open(os.path.join(args.split_dir, f"stage{split[-1]}", f"forget{split}.json"), encoding="utf-8") as f:
        splits["forget"] = json.load(f)
    with open(os.path.join(args.split_dir, f"stage{split[-1]}", f"forget{split}_NU.json"), encoding="utf-8") as f:
        splits["forget_NU"] = json.load(f)
    with open(os.path.join(args.split_dir, f"stage{split[-1]}", f"retain_perturbed.json"), encoding="utf-8") as f:
        splits["retain"] = json.load(f)
    with open(os.path.join(args.split_dir, f"stage{split[-1]}", f"real_authors.json"), encoding="utf-8") as f:
        splits["real_authors"] = json.load(f)
    with open(os.path.join(args.split_dir, f"stage{split[-1]}", f"world_facts.json"), encoding="utf-8") as f:
        splits["world_facts"] = json.load(f)

    with open(os.path.join(args.split_dir, f"stage{split[-1]}", f"forget{split}.json"), encoding="utf-8") as f:
        forget_split = json.load(f)
        id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in forget_split}

    MAPPING_PATH = Path(args.split_dir) / f"stage{split[-1]}" / f"TOFU_to_forget{split}_top3_{ablation_files[ablation]}.json"
    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)

    result: Dict[str,Dict] = {}
    for name, ds in splits.items():
        agg= eval_subset(name, ds, id2question,
                                  ID_MAP,
                                  batch_size=args.batch_size, )
        result[name] = {"metrics": agg}
        print(f"[{name}] {json.dumps(agg, indent=2, ensure_ascii=False)}")

    out = os.path.join(args.output_dir, "tofu_eval_results.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {out}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
