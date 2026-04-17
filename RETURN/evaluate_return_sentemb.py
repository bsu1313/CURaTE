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

REFUSAL_PATH = Path("../refusal_answer.json")  
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))

def get_available_cache_dir():
    preferred = Path("/home/work/data/")

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
        question_start_token = "<s>[INST] "
        question_end_token = " [/INST]"
    else:
        raise ValueError('Please provide llama model')
    # print("wrapped prompt: ", f"{question_start_token}{p}{question_end_token}")
    return f"{question_start_token}{p}{question_end_token}"
def build_llama2_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,   
        add_generation_prompt=True  
    )
    return prompt

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _mean(x: List[float]): return float(np.mean(x)) if x else 0.0

def truth_ratio(tp: float, fp: List[float]):
    return (np.mean(fp)+1e-12)/(tp+1e-12)

def acc_contains(pred, truth):
    return int(bool(re.search(re.escape(truth), pred, re.I)))


def load_model(base, ds_cfg, cache_dir, dtype=torch.float16):
    cfg = transformers.AutoConfig.from_pretrained(base)
    cfg.tp_size = 1          # disable HF-TP

    model = AutoModelForCausalLM.from_pretrained(
        base, config=cfg, torch_dtype=dtype,
        low_cpu_mem_usage=True, device_map=None,
        cache_dir=cache_dir
    )

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
                              # max_new_tokens=32,
                              max_length = 256,
                              do_sample=False,
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

# def eval_subset(model, tok, model_name, name, ds, id2question, ID_MAP, batch_size=4):
def eval_subset(model, tok, model_name, name, ds, ID_MAP, batch_size=4):
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
        prompts_1, questions_1, preds_1, correct_1, incorrect_1 = [], [], [], [], []
        for item in batch:
            # print("item: ", item)
            if name == "forget":
                question = item["paraphrased_instruction"]
                answer = item["gold_answer"]
            elif name == "near_utility":
                question = item["contrastive_instruction"]
                answer = item["contrastive_answer"]
            elif name == "winogrande":
                question = build_WD_prompt(
                    item["sentence"], item["option1"], item["option2"]
                )
                # print("item[answer]: ", item["answer"])
                # print("item[answer]==1: ", item["answer"]=="1")
                if item["answer"] == "1":
                    answer = item["option1"]
                    incorrect_answer = item["option2"]
                else:
                    answer = item["option2"]
                    incorrect_answer = item["option1"]
                incorrect_1.append(incorrect_answer)

            else:
                question = item["question"]
                answer = item["gold_answer"]
            # print("question: ", question)
            questions_1.append(question)
            # ref_q = mapped_question(item["id"], id2question, ID_MAP)
            cos_sim = mapped_cossim(item["id"], ID_MAP)
            # print("cos_sim: ", cos_sim)
            max_cos_sim = max(float(x) for x in cos_sim) if cos_sim else 0.0

            if max_cos_sim > 0.9:
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

            # prompts_1.append(build_llama2_prompt(question, tok))
            prompts_1.append(wrap_prompt(question, model_name.lower()))
            correct_1.append(answer)
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
            if name == "winogrande":
                inc = incorrect_1[i]
                score = 1 if (ans_gt.lower() in gen.lower() and inc.lower() not in gen.lower()) else 0
                # print("question: ", questions_1[i])
                # print("ans_gt: ", ans_gt)
                # print("inc: ", inc)
                # print("gen: ", gen)
                # print("score: ", score)

                metrics["acc"].append(score)
                samples.append({
                    "question": questions_1[i],
                    "truth": ans_gt,
                    "generated": gen,
                    "acc": score,
                })
            else:
                rouge_rec = rouge.score(ans_gt, gen)["rougeL"].recall
                # print("rouge_rec: ", rouge_rec)
                metrics["rougeL"].append(rouge_rec)
                samples.append({
                    "question": questions_1[i],
                    "truth": ans_gt,
                    "generated": gen,
                    "rougeL_recall": rouge_rec,
                })
    agg = {k: _mean(v) for k, v in metrics.items()}
    agg["positives"] = par_positives
    agg["negatives"] = par_negatives
    return agg, samples


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
    # ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
    # ap.add_argument("--base_model", default="../models/Llama-3.2-1B-Instruct")
    ap.add_argument("--ds_config", default="../ds_config.json")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", default="./eval_results")
    ap.add_argument("--cache_dir", default=get_available_cache_dir())
    ap.add_argument("--local_rank", type=int, default=-1, help="(set by deepspeed)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if "1B" in args.base_model:
        split_dir = "Meta-Llama-3.2-1B-Instruct_dataset"
    elif "7b" in args.base_model:
        split_dir = "Meta-Llama-2-7B-chat_dataset"

    model, tok = load_model(args.base_model,
                            args.ds_config,
                            args.cache_dir)


    # model_dir = "mpnet_contrastive_model"
    # sent_model = SentenceTransformer(model_dir)

    splits = {}
    split = "0" # 0,1,2,3,4,5,6,7,8,9
    with open(os.path.join(split_dir, f"stage_{split}_forget_paraphrased.json"), encoding="utf-8") as f:
        splits["forget"] = json.load(f)
    with open(os.path.join(split_dir, f"stage_{split}_retain_used.json"), encoding="utf-8") as f:
        splits["retain_used"] = json.load(f)
    with open(os.path.join(split_dir, f"stage_{split}_retain_not_used.json"), encoding="utf-8") as f:
        splits["retain_not_used"] = json.load(f)
    with open(os.path.join(split_dir, "non_target.json"), encoding="utf-8") as f:
        splits["non_target"] = json.load(f)
    with open(os.path.join(split_dir, f"stage_{split}_near_utility.json"), encoding="utf-8") as f:
        splits["near_utility"] = json.load(f)
    with open(os.path.join(split_dir, "winogrande_xs_validation.json"), encoding="utf-8") as f:
        splits["winogrande"] = json.load(f)

    # with open(os.path.join(split_dir, f"stage_{split}_forget_paraphrased.json"), encoding="utf-8") as f:
    #     forget_split = json.load(f)
    #     id2question: dict[int, str] = {ex["id"]: ex["paraphrased_instruction"] for ex in forget_split}
    # with open(os.path.join(split_dir, f"stage_{split}_forget.json"), encoding="utf-8") as f:
    #     forget_split = json.load(f)
    #     id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in forget_split}

    MAPPING_PATH = Path(split_dir) / f"RETURN_stage_{split}_top3.json"
    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)

    result: Dict[str,Dict] = {}
    for name, ds in splits.items():
        agg, detail = eval_subset(model, tok, args.base_model, name, ds, #id2question,
                                  ID_MAP,
                                  batch_size=args.batch_size, )
        result[name] = {"metrics": agg, "samples": detail}
        print(f"[{name}] {json.dumps(agg, indent=2, ensure_ascii=False)}")
    final_metrics = {name: res["metrics"] for name, res in result.items()}
    print("\n==== Final Aggregated Metrics ====")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))

    out = os.path.join(args.output_dir, "return_eval_results.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {out}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
