#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified TOFU evaluation
-----------------------
* base-model + LoRA 로드 (DeepSpeed inference)
* seen / unseen / retain / real_authors / world_facts 5-split 평가
* 결과를 JSON(+샘플)로 저장
"""

import os, sys, json, math, random, argparse, tqdm, re
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
import deepspeed
import transformers
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
# from datasets import Dataset
# --------------------------------------------------------------------------
# 프롬프트 템플릿
# --------------------------------------------------------------------------
from conversation import get_conv_template        # 💡 경로 확인!

from sentence_transformers import SentenceTransformer, util
from pathlib import Path
# from train_classifier0 import BinaryClassifier
import random

REFUSAL_PATH = Path("../refusal_answer.json")   # ← 실제 파일명/경로
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))

# class BinaryClassifier(nn.Module):
#     def __init__(self, input_dim):
#         super(BinaryClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return self.sigmoid(x)

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Increase the number of units
        self.fc2 = nn.Linear(512, 256)       # Add an additional layer
        self.fc3 = nn.Linear(256, 1)         # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Additional activation function
        x = self.fc3(x)
        return self.sigmoid(x)


def get_available_cache_dir():
    preferred = Path("/home/david/.cache")
    fallback = Path("/home/plowcow/.cache")

    if preferred.exists() and os.access(preferred, os.W_OK):
        return str(preferred)
    else:
        return str(fallback)

# ① 매핑용 모델
map_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

def build_forget_index(forget_ds, bs=128):
    """forget01_perturbed 의 question → embedding 인덱스(one-shot)"""
    f_texts = forget_ds["question"]
    f_embs  = map_model.encode(
        f_texts, batch_size=bs, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=True)
    return f_texts, f_embs        # list[str], Tensor[N,768]
# --------------------------------------------------------------------------
# 🔥 NEW: raw_q → forget_q 매핑 헬퍼
# --------------------------------------------------------------------------
def match_forget_questions(raw_qs, f_texts, f_embs,
                           mapping: Dict[str,str]) -> List[str]:
    """batch 단위 raw_qs 에 대해 best-match forget_q 반환 & 매핑 업데이트"""
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
        # 유사도 (dot score == cosine when normalized)
        sim = rq_embs @ f_embs.T                # [M, N]
        best_idx = sim.argmax(dim=1).tolist()
        for q, idx in zip(need_compute, best_idx):
            mapping[q] = f_texts[idx]           # 캐시에 추가

    return [mapping[q] for q in raw_qs]

def mapped_question(origin_id: int, id2question, ID_MAP) -> List[str]:
    """
    Args:
        origin_id : 현재 예시의 id  (e.g. 5)
        key       : "paraphrased" or "contrastive"
    Returns:
        매핑된 id( top-3 의 첫 번째 )에 대응하는 question 문자열
        (없으면 원본 question 을 그대로 반환)
    """
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
# def build_llama2_prompt(question: str,
#                         forgotten_info: str = "") -> str:
#     # conv = get_conv_template("llama-2")
#     # conv.append_message(conv.roles[0], question)   # user
#     # conv.append_message(conv.roles[1], None)         # assistant
#     # final_prompt = conv.get_prompt()
#
#     final_prompt = "[INST] " + question + " [/INST]"
#     return final_prompt
# def build_llama2_prompt(question: str,
#                         forgotten_info: str = "") -> str:
#     """Llama-2 chat 템플릿으로 감싼다."""
#     if forgotten_info == "":
#         forgotten_info = (
#             "Basil Mahfouz Al-Kuwaiti"
#         )
#
#
#     input_text = (
#         "Based on the [Forgotten Information], decide whether to answer or "
#         "refuse to answer the [Query]. Then provide an appropriate response "
#         "accordingly.\n\n"
#         f"[Forgotten Information]:\n{forgotten_info}\n\n"
#         f"[Query]:\n{question}"
#     )
#
#     conv = get_conv_template("llama-2")
#     conv.set_system_message(
#         "You are a helpful, respectful and honest assistant."
#     )
#     conv.append_message(conv.roles[0], input_text)   # user
#     conv.append_message(conv.roles[1], None)         # assistant
#     return conv.get_prompt()

# --------------------------------------------------------------------------
# 평가 지표
# --------------------------------------------------------------------------
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
# st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def _mean(x: List[float]): return float(np.mean(x)) if x else 0.0

def truth_ratio(tp: float, fp: List[float]):
    return (np.mean(fp)+1e-12)/(tp+1e-12)

def acc_contains(pred, truth):
    return int(bool(re.search(re.escape(truth), pred, re.I)))

# def acc_semantic(pred, truth, falses):
#     emb_p = st_model.encode(pred,  convert_to_tensor=True)
#     emb_t = st_model.encode(truth, convert_to_tensor=True)
#     sims  = [util.pytorch_cos_sim(emb_p, emb_t)]
#     for f in falses:
#         sims.append(util.pytorch_cos_sim(
#             emb_p, st_model.encode(f, convert_to_tensor=True)))
#     return int(torch.argmax(torch.tensor(sims)) == 0)

# --------------------------------------------------------------------------
# 모델 로드
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 생성
# --------------------------------------------------------------------------

def postprocess_completion(comp: str) -> str:
    """
    1) [Reason] 포함 뒷부분 제거
    2) 두 줄 공백이 나오더라도 내용이 비어 있으면 버리지 않기
    3) 완전히 비면 첫 번째 실질적인 non-empty line을 살려 줌
    """
    # ① [Reason] 이후 잘라내기 (토큰 포함 X)
    cut = comp.find("[Reason]")
    if cut != -1:
        comp = comp[:cut]

    # ② 첫 번째 문단만 가져오되, 문단이 비어 있으면 넘김
    # paras = [p.strip() for p in comp.split("\n\n") if p.strip()]
    # if paras:
    #     comp = paras[0]

    # ③ 그래도 비어 있다면 한 줄짜리라도 반환
    return comp.strip()


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

# --------------------------------------------------------------------------
# perplexity-based 확률
# --------------------------------------------------------------------------
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

# def score_answer_prob(model, tok, question,
#                       truth_ans, falses):
#     prompt = build_llama2_prompt(question)
#     p_true = seq_prob(model, tok, prompt+truth_ans)
#     p_false = [seq_prob(model, tok, prompt+f) for f in falses] if falses else []
#     return p_true, p_false

# --------------------------------------------------------------------------
# subset 평가
# --------------------------------------------------------------------------
def predict(texts, tokenizer, model, max_length=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []

    for text in texts:
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0][pred_class].item()

        predictions.append({
            "text": text,
            "pred_class": pred_class,
            "probability": pred_prob
        })

    return predictions

def eval_subset(model, tok, sent_model, bc_model, model_name, name, ds, forget_data, ID_MAP, batch_size=4):
    # dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in forget_data}
    # print("id2question: ", id2question)

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

            # ## use sentence embedding model
            match = False
            q_emb = sent_model.encode(question, convert_to_tensor=True)
            for f_info in ref_q:
                f_emb = sent_model.encode(f_info, convert_to_tensor=True)
                # input = torch.cat([q_emb, f_emb], dim=0).unsqueeze(0)  # [1, 2, 768]
                input = torch.cat((f_emb, q_emb), dim=0).unsqueeze(0).to("cuda")
                match_prob = bc_model(input)  # [1, 2]
                if match_prob.item() > 0.5:  # threshold for similarity
                    match = True

            ## use mapped cosine similarity
            # cos_sim = mapped_cossim(item["id"], ID_MAP)
            # max_cos_sim = max(float(x) for x in cos_sim) if cos_sim else 0.0
            #
            # if max_cos_sim > 0.8:
            #     match = True
            # else:
            #     match = False

            if not match:
                preds_1.append(0)
                par_negatives += 1
            else:
                preds_1.append(1)
                par_positives += 1

            # prompts_1.append(build_llama2_prompt(question, tok))
            prompts_1.append(wrap_prompt(question, model_name.lower()))
            correct_1.append(item["answer"])
            # print("prompts_1: ", prompts_1)
            # print("correct_1: ", correct_1)

        gens_1 = batched_generate(model, tok, prompts_1)
        # print("gens_1 before: ", gens_1)

        for i, pred in enumerate(preds_1):
            if pred == 1:
                gens_1[i] = random.choice(REF_PHRASES)
            elif pred == 0:
                gens_1[i] = gens_1[i].strip()
            else:
                raise ValueError(f"Unexpected prediction class: {pred}")
        # print("gens_1 after: ", gens_1)

        for i, gen in enumerate(gens_1):
            ans_gt = correct_1[i]
            # print("ans_gt: ", ans_gt)
            # print("gen: ", gen)
            rouge_rec = rouge.score(ans_gt, gen)["rougeL"].recall
            # print("rouge_rec: ", rouge_rec)

            metrics["rougeL"].append(rouge_rec)
            # metrics["acc"].append(acc)

            samples.append({
                "question": questions_1[i],
                "truth": ans_gt,
                "generated": gen,
                # "truth_prob": p_true,
                # "false_probs": p_false,
                # "truth_ratio": ratio,
                "rougeL_recall": rouge_rec,
                # "acc": acc,
            })
        # sys.exit()
    agg = {k: _mean(v) for k, v in metrics.items()}
    agg["positives"] = par_positives
    agg["negatives"] = par_negatives
    return agg, samples

# --------------------------------------------------------------------------
# 데이터 split 로드
# --------------------------------------------------------------------------
def load_split(name, cache):
    return load_dataset("locuslab/TOFU", name,
                        cache_dir=cache, split="train")
    # return Dataset.load_from_disk("/home/work/seyun_workspace/cache_LTE/TOFU/"+ name)
    

def get_seen_unseen(ds, ratio=0.8, seed=1000):
    random.seed(seed)
    idx_seen = random.sample(range(len(ds)), int(len(ds)*ratio))
    idx_unseen = list(set(range(len(ds))) - set(idx_seen))
    return ds.select(idx_seen), ds.select(idx_unseen)

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--base_model", required=True)
    # ap.add_argument("--base_model", default="open-unlearning/tofu_Llama-2-7b-chat-hf_full")
    ap.add_argument("--base_model", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    # ap.add_argument("--lora_path",  required=True)
    # ap.add_argument("--ds_config",  required=True)
    ap.add_argument("--ds_config", default="ds_config.json")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", default="./eval_results")
    # ap.add_argument("--cache_dir",  default="/home/work/seyun_workspace/cache_LTE/")
    # ap.add_argument("--cache_dir", default="/home/david/.cache/")
    ap.add_argument("--cache_dir", default=get_available_cache_dir())
    ap.add_argument("--local_rank", type=int, default=-1, help="(set by deepspeed)")
    # ap.add_argument("--split_dir", default="TOFU_continual")
    ap.add_argument("--split_dir", default="TOFU_continual_new")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 모델
    model, tok = load_model(args.base_model,
                            # args.lora_path,
                            args.ds_config,
                            args.cache_dir)


    model_dir = "../mpnet_contrastive_model"
    # sent_model = SentenceTransformer(model_dir)
    # sent_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    # sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # sent_model = SentenceTransformer('bert-base-nli-mean-tokens')
    # sent_model = SentenceTransformer('all-mpnet-base-v2')
    # sent_model = SentenceTransformer('all-MiniLM-L6-v2')
    # sent_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    sent_model = SentenceTransformer('sentence-transformers-testing/stsb-bert-tiny-safetensors')

    splits = {}
    split = "1"
    with open(os.path.join(args.split_dir, f"forget{split}", f"forget{split}.json"), encoding="utf-8") as f:
        splits["forget"] = json.load(f)
    with open(os.path.join(args.split_dir, f"forget{split}", f"retain_perturbed.json"), encoding="utf-8") as f:
        splits["retain"] = json.load(f)
    with open(os.path.join(args.split_dir, f"forget{split}", f"real_authors.json"), encoding="utf-8") as f:
        splits["real_authors"] = json.load(f)
    with open(os.path.join(args.split_dir, f"forget{split}", f"world_facts.json"), encoding="utf-8") as f:
        splits["world_facts"] = json.load(f)

    MAPPING_PATH = Path(args.split_dir) / f"forget{split}" / f"TOFU_to_forget{split}_top3.json"
    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)

    print("HERE")
    # 데이터
    # forget_per = load_split("forget01_perturbed", args.cache_dir)
    print("END")
    
    # seen, unseen = get_seen_unseen(forget_per)

    # f_texts, f_embs = build_forget_index(forget_per)
    # map_path = os.path.join(".", "raw2forget_map.json")
    # q2f_map = json.load(open(map_path)) if os.path.exists(map_path) else {}
    #
    #
    # splits = {
    #     "forget"        : forget_per.shuffle(seed=42),
    #     # "unseen"      : unseen,
    #     "retain"      : load_split("retain_perturbed",       args.cache_dir),
    #     "real_authors": load_split("real_authors_perturbed", args.cache_dir),
    #     "world_facts" : load_split("world_facts_perturbed",  args.cache_dir),
    # }

    with open("binary_classifier/config.json", "r") as f:
        bc_config = json.load(f)
    input_dim = bc_config["input_dim"]
    # Initialize model
    bc_model = BinaryClassifier(input_dim).to("cuda")
    # Load weights
    bc_model.load_state_dict(torch.load("binary_classifier/binary_classifier.pth"))
    bc_model.eval()


    result: Dict[str,Dict] = {}
    for name, ds in splits.items():
        agg, detail = eval_subset(model, tok, sent_model, bc_model, args.base_model, name, ds, splits["forget"],
                                  ID_MAP,
                                  batch_size=args.batch_size, )
        result[name] = {"metrics": agg, "samples": detail}
        print(f"[{name}] {json.dumps(agg, indent=2, ensure_ascii=False)}")

    out = os.path.join(args.output_dir, "tofu_eval_results.json")

    # lora_name = os.path.basename(os.path.normpath(args.lora_path))
    # out = os.path.join(
    #     args.output_dir,
    #     f"tofu_eval_results_{lora_name}.json"
    # )

    # with open(map_path, "w", encoding="utf-8") as f:
    #     json.dump(q2f_map, f, indent=2, ensure_ascii=False)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {out}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
