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
from typing import List, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
import deepspeed
import transformers

from rouge_score import rouge_scorer
from datasets import Dataset
# --------------------------------------------------------------------------
# 프롬프트 템플릿
# --------------------------------------------------------------------------
from conversation import get_conv_template        # 💡 경로 확인!

from sentence_transformers import SentenceTransformer, util
from pathlib import Path

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


def build_llama2_prompt(question: str,
                        forgotten_info: str = "") -> str:
    # conv = get_conv_template("llama-2")
    # conv.append_message(conv.roles[0], question)   # user
    # conv.append_message(conv.roles[1], None)         # assistant
    # final_prompt = conv.get_prompt()

    final_prompt = "[INST] " + question + " [\INST]"
    return final_prompt

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
    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        # base,
        use_fast=False, padding_side="left",
        cache_dir=cache_dir
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


def batched_generate(model, tok, prompts, max_new=128):
    # print("prompts: ", prompts)
    inputs = tok(prompts, return_tensors="pt",
                 padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        outs = model.generate(**inputs,
                              # max_new_tokens=256,
                              max_length = 200,
                              do_sample=False,
                              # min_new_tokens=4,
                              eos_token_id=tok.eos_token_id,
                              use_cache=False)

    results = []
    for ids in outs:
        # ① special token을 살린 채 풀 디코드
        full = tok.decode(ids,
                          skip_special_tokens=True,
                          clean_up_tokenization_spaces=False)

        # print("full: ", full)
        comp = full.split("[/INST]", 1)[-1].strip()
        # print("comp1: ", comp)
        comp = postprocess_completion(comp)
        # print("comp2: ", comp)
        results.append(comp)

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

def eval_subset(model, tok, name, ds, f_texts, f_embs, q2f_map, roberta_model, roberta_tok, batch_size=4):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    metrics = {k:[] for k in
               ("truth_ratio","truth_prob","rougeL","acc")}
    samples = []
    positives = 0
    negatives = 0

    for batch in tqdm.tqdm(dl, desc=f"Eval {name}"):
        has_pq  = "paraphrased_question" in batch
        # has_pa  = "paraphrased_answer"  in batch
        # print("batch.keys(): ", batch.keys()) # 'question', 'answer'
        # print("batch sample: ", batch)
        # print("has_pq: ", has_pq) # False
        # if has_pq:
        #     sys.exit()

        # 질문 리스트
        # raw_qs  = batch["paraphrased_question"] if has_pq else batch["question"]
        raw_qs = batch["question"]
        

        ## TODO ##
        forget_infos = match_forget_questions(raw_qs, f_texts, f_embs, q2f_map)
        # print("raw qs sample: ", raw_qs[:5])
        # print("forget_infos sample: ", forget_infos[:5])
        prompts = [build_llama2_prompt(q, forgotten_info=f_info)
                        for q, f_info in zip(raw_qs, forget_infos)]
        roberta_prompts = ["[Forgotten Information]:\n" + f_info + "\n\n[Query]:\n" + q for q, f_info in zip(raw_qs, forget_infos)]
        # print("roberta_prompts sample: ", repr(roberta_prompts[0]))
        # sys.exit()
        predictions = predict(roberta_prompts, roberta_tok, roberta_model)
        preds = [p["pred_class"] for p in predictions]
        positives += preds.count(1)
        negatives += preds.count(0)
        # for q, f_info, p in zip(raw_qs, forget_infos, preds):
        #     print(f"Forgotten Info: {f_info}\nQuery: {q}\nPrediction: {p}\n")

        # for rprompt, p in zip(roberta_prompts, preds):
        #     print(f"Roberta Prompt: {rprompt}\nPrediction: {p}\n")

        #prompts = [build_llama2_prompt(qs) for qs in raw_qs]
        gens    = batched_generate(model, tok, prompts)
        # print("sample GT answers: ", gens[:5])

        for i, pred in enumerate(preds):
            if pred == 1:
                gens[i] = "I don't have information about that."
                # gens[i] = gens[i].strip()
            elif pred == 0:
                gens[i] = gens[i].strip()
            else:
                raise ValueError(f"Unexpected prediction class: {pred}")

        # print("sample generated answers: ", gens[:10])
        # sys.exit()


        for i, gen in enumerate(gens):
            raw_q   = raw_qs[i]
            ans_gt  = batch["answer"][i]
            # para_gt = (batch["paraphrased_answer"][i]
            #            if has_pa and i < len(batch["paraphrased_answer"])
            #            else ans_gt)

            # false list 안전 추출
            # if "perturbed_answer" in batch and i < len(batch["perturbed_answer"]):
            #     falses = batch["perturbed_answer"][i] or []
            # else:
            #     falses = []

            # p_true, p_false = score_answer_prob(
            #     model, tok, raw_q, para_gt, falses)
            # ratio = truth_ratio(p_true, p_false)

            # acc = (acc_semantic(gen, para_gt, falses)
            #        if has_pa else acc_contains(gen, ans_gt))

            rouge_rec = rouge.score(ans_gt, gen)["rougeL"].recall

            # 기록
            # metrics["truth_ratio"].append(ratio)
            # metrics["truth_prob"].append(p_true)
            metrics["rougeL"].append(rouge_rec)
            # metrics["acc"].append(acc)

            samples.append({
                "question": raw_q,
                "truth": ans_gt,
                "generated": gen,
                # "truth_prob": p_true,
                # "false_probs": p_false,
                # "truth_ratio": ratio,
                "rougeL_recall": rouge_rec,
                # "acc": acc,
            })

    agg = {k:_mean(v) for k,v in metrics.items()}
    agg["positives"] = positives
    agg["negatives"] = negatives
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
    ap.add_argument("--base_model", default="open-unlearning/tofu_Llama-2-7b-chat-hf_full")
    # ap.add_argument("--base_model", default="open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    # ap.add_argument("--lora_path",  required=True)
    # ap.add_argument("--ds_config",  required=True)
    ap.add_argument("--ds_config", default="ds_config.json")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", default="./eval_results")
    # ap.add_argument("--cache_dir",  default="/home/work/seyun_workspace/cache_LTE/")
    # ap.add_argument("--cache_dir", default="/home/david/.cache/")
    ap.add_argument("--cache_dir", default=get_available_cache_dir())
    ap.add_argument("--local_rank", type=int, default=-1,
                    help="(set by deepspeed)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 모델
    model, tok = load_model(args.base_model,
                            # args.lora_path,
                            args.ds_config,
                            args.cache_dir)

    model_dir = "roberta_features_Aprime_classifier"
    roberta_tok = RobertaTokenizer.from_pretrained(model_dir)
    roberta_model = RobertaForSequenceClassification.from_pretrained(model_dir)
    roberta_model.eval()
    print("HERE")
    # 데이터
    forget_per = load_split("forget01_perturbed", args.cache_dir)
    print("END")
    
    seen, unseen = get_seen_unseen(forget_per)

    f_texts, f_embs = build_forget_index(forget_per)
    map_path = os.path.join(".", "raw2forget_map.json")
    q2f_map = json.load(open(map_path)) if os.path.exists(map_path) else {}
    

    splits = {
        "forget"        : forget_per.shuffle(seed=42),
        # "unseen"      : unseen,
        "retain"      : load_split("retain_perturbed",       args.cache_dir),
        "real_authors": load_split("real_authors_perturbed", args.cache_dir),
        "world_facts" : load_split("world_facts_perturbed",  args.cache_dir),
    }

    # splits = {
    #     "forget"      : load_split("forget01",       args.cache_dir),
    #     "retain"      : load_split("retain99",       args.cache_dir),
    #     "real_authors": load_split("real_authors", args.cache_dir),
    #     "world_facts" : load_split("world_facts",  args.cache_dir),
    # }

    result: Dict[str,Dict] = {}
    for name, ds in splits.items():
        agg, detail = eval_subset(model, tok, name, ds,
                                  f_texts, f_embs, q2f_map,
                                  roberta_model, roberta_tok,
                                  batch_size=args.batch_size, )
        result[name] = {"metrics": agg, "samples": detail}
        print(f"[{name}] {json.dumps(agg, indent=2, ensure_ascii=False)}")

    out = os.path.join(args.output_dir, "tofu_eval_results.json")

    # lora_name = os.path.basename(os.path.normpath(args.lora_path))
    # out = os.path.join(
    #     args.output_dir,
    #     f"tofu_eval_results_{lora_name}.json"
    # )

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(q2f_map, f, indent=2, ensure_ascii=False)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {out}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
