# -*- coding: utf-8 -*-
"""
Evaluation script adapted from the original TruthfulQA evaluator.

Key changes
-----------
1. **Dataset:** Now uses the CommonsenseQA dataset hosted at
   https://huggingface.co/datasets/tau/commonsense_qa.
2. **ID Mapping:** If an `id_mapping` JSON is provided it will be loaded, but the
   evaluation itself does **not** rely on paraphrased/contrastive questions – it
   directly evaluates the model on standard CommonsenseQA items.
3. **Metrics:** Computes **exact‑match accuracy** between the predicted choice
   label (`A`‒`E`) and the gold `answerKey`, and also reports macro‑averaged
   Rouge‑L recall against the gold choice text (optional).
4. **Prompts & Parsing:** Adds a compact multiple‑choice prompt that asks the
   model for the correct letter. Prediction parsing grabs the first capital
   letter A‑E found in the generation.

Most other utilities (model loading, batched generation, DeepSpeed etc.) are
kept as‑is to maximise code reuse.
"""

import os, sys, json, math, argparse, re, random, tqdm
from typing import List, Dict, Any, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import deepspeed, transformers

from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from torch.utils.data.dataloader import default_collate
from pathlib import Path

from datasets import load_dataset  # NEW

# Prompt template helpers (unchanged)
from conversation import get_conv_template  # ensure import path is correct

# ─────────────────────────────────────────────────────────────────────────────
# Globals & mapping placeholders (optional)
# ─────────────────────────────────────────────────────────────────────────────
ID_MAPPING_PATH: Path | None = None  # TODO
ID_MAP: Dict[str, Dict[str, Any]] | None = None   # TODO

# If mapping json is supplied we load it, but evaluation logic does not depend
# on it for CommonsenseQA.

# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────



def format_forgotten_info(questions: List[str]) -> str:
    """1, 2, 3 … 형태로 개행 구분 포매팅."""
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

def build_commonsense_prompt(question: str, forgotten_info: str, choices: List[Tuple[str, str]]) -> str:
    """Construct a simple multiple‑choice prompt for the model."""
    choice_block = "\n".join([f"{label}. {text}" for label, text in choices])

    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a helpful, respectful and honest assistant.")
    # user_msg = (
    #     "Please respond with only the letter of the correct answer (A, B, C, D, or E) with no explanation."
    #     f"Question: {question}\n\nChoices:\n{choice_block}\n\n"
    #     "Answer: \n"
    # )
    
    
    user_msg = (
        f"{question}\n\nChoices:\n{choice_block}\n\n"
        # "Please respond with only the letter of the correct answer (A, B, C, D, or E) with no explanation."
    )
    
    
    input_text = f"Based on the [Forgotten Information], decide whether to answer or refuse to answer the [Query]. Then provide an appropriate response accordingly. If you choose to refuse, include the corresponding [Reason] for your decision.\n\n[Forgotten Information]:\n{forgotten_info}\n\n[Query]:\n{user_msg}"

    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def parse_letter(pred: str) -> str | None:
    """Returns first capital letter A‑E found in *pred*, else None."""
    m = re.search(r"[A-E]", pred)
    return m.group(0) if m else None

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (Rouge kept for optional analysis)
# ─────────────────────────────────────────────────────────────────────────────
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (identical to original)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(base: str, lora: str, ds_cfg: str, dtype=torch.float16):
    cfg = transformers.AutoConfig.from_pretrained(base)
    cfg.tp_size = 1

    model = AutoModelForCausalLM.from_pretrained(
        base,
        config=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir="/home/work/hangyul/seyun_workspace/cache_LTE/",
    )
    model = PeftModel.from_pretrained(model, lora).merge_and_unload()

    engine = deepspeed.init_inference(
        model,
        dtype=dtype,
        kernel_inject=False,
        replace_method="auto",
        config=json.load(open(ds_cfg)),
    )

    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        use_fast=False,
        padding_side="left",
        cache_dir="/home/work/hangyul/seyun_workspace/cache_LTE/",
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return engine.module, tok

# ─────────────────────────────────────────────────────────────────────────────
# Batched generation (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def batched_generate(model, tok, prompts: List[str]) -> List[str]:
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            min_new_tokens=4,
            eos_token_id=tok.eos_token_id,
            use_cache=False,
        )

    results = []
    for ids in outs:
        full = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Everything after [/INST] is the model's answer in the llama‑2 template
        comp = full.split("[/INST]", 1)[-1].strip()
        results.append(comp)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# CommonsenseQA evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_commonsenseqa(model, tok, split: str = "validation", batch_size: int = 4):
    ds = load_dataset("tau/commonsense_qa", split=split)
    ds.save_to_disk("/home/work/seyun_workspace/cache_LTE/commonsense_qa")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x  )

    preds, gold_labels, gold_texts, questions = [], [], [], []
    rouge_recalls = []                       # ➊ 신규 리스트

    for batch in tqdm.tqdm(dl, desc=f"Evaluating {split}"):
        prompts = []
        batch_labels = []
        batch_gold_texts = []
        batch_questions = []

        for ex in batch:
            labels = ex["choices"]["label"]        # ["A", "B", ...]
            texts  = ex["choices"]["text"]         # ["sand", ...]
            choices = list(zip(labels, texts))
            
            n_forgotten =3
            if ID_MAP is not None and n_forgotten > 0:
                if ex["question"] not in ID_MAP:
                    raise KeyError(
                        f"[ID_MAP]에 해당 질문이 없습니다: «{ex['question']}»"
                    )
                cand_list = ID_MAP[ex["question"]][:n_forgotten]
            else:
                cand_list = []

            if len(cand_list) == 1:
                forgotten_info = cand_list[0]          # 그대로 사용
            else:
                forgotten_info = format_forgotten_info(cand_list) if cand_list else ""
            

            prompts.append(build_commonsense_prompt(ex["question"], forgotten_info, choices))
            batch_labels.append(ex["answerKey"])
            batch_questions.append(ex["question"].strip())

            # map answerKey -> full text for qualitative logs
            gold_text = dict(zip(labels, texts))[ex["answerKey"]]
            batch_gold_texts.append(gold_text)

        gens = batched_generate(model, tok, prompts)
    
    
        for pred, gold in zip(gens, batch_gold_texts):
            rouge_dict = rouge.score(gold, pred)         # {'rougeL': Score(...)}
            rouge_recalls.append(rouge_dict["rougeL"].recall)

        preds.extend(gens)
        gold_labels.extend(batch_labels)
        gold_texts.extend(batch_gold_texts)
        questions.extend(batch_questions)

    # Accuracy
    correct = sum(p == g for p, g in zip(preds, gold_labels))
    acc = correct / len(preds)
    rougeL_recall = float(np.mean(rouge_recalls)) 

    
    aggregate = {
        "accuracy": acc,
        "rougeL_recall": rougeL_recall,
    }
    # Prepare per‑sample log
    samples = [
        {
            "question": q,
            "gold_label": g,
            "gold_text": gt,
            "prediction": p,
            "rougeL_recall": r,
            "correct": int(p == g),
        }
        for q, g, gt, p, r in zip(questions, gold_labels, gold_texts, preds, rouge_recalls)
    ]
    return aggregate, samples

# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate model on CommonsenseQA")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--ds_config", required=True)
    ap.add_argument("--output_dir", default="./eval_results_commonsense")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--local_rank", type=int, default=-1)
    ap.add_argument("--id_mapping_json", default=None, help="(Optional) new id mapping JSON")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global ID_MAPPING_PATH, ID_MAP
    if args.id_mapping_json:
        ID_MAPPING_PATH = Path(args.id_mapping_json)
        ID_MAP = json.loads(ID_MAPPING_PATH.read_text(encoding="utf-8"))

    # Load model
    model, tok = load_model(args.base_model, args.lora_path, args.ds_config)

    # Evaluate
    aggregate, samples = eval_commonsenseqa(
        model, tok, split=args.split, batch_size=args.batch_size
    )

    # Save
    out_samples = Path(args.output_dir) / "commonsenseqa_samples.json"
    out_agg = Path(args.output_dir) / "commonsenseqa_metrics.json"
    out_samples.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    out_agg.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n✅ Saved per‑sample results to", out_samples)
    print("✅ Saved aggregate metrics to", out_agg)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
