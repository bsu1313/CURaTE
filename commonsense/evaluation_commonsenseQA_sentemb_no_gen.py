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
from transformers import RobertaTokenizer, RobertaForSequenceClassification

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
import random

from datasets import load_dataset  # NEW

# ─────────────────────────────────────────────────────────────────────────────
# Globals & mapping placeholders (optional)
# ─────────────────────────────────────────────────────────────────────────────

REFUSAL_PATH = Path("../refusal_answer.json")   # ← 실제 파일명/경로
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))

def mapped_question(origin_id: int, key: str, id2question, ID_MAP) -> List[str]:
    """
    Args:
        origin_id : 현재 예시의 id  (e.g. 5)
        key       : "paraphrased" or "contrastive"
    Returns:
        매핑된 id( top-3 의 첫 번째 )에 대응하는 question 문자열
        (없으면 원본 question 을 그대로 반환)
    """
    try:
        mapped_ids = ID_MAP[str(origin_id)][f"{key}_top3_ids"]
        return [id2question[mid] for mid in mapped_ids if mid in id2question]
    except (KeyError, IndexError):
        return id2question[origin_id]

def mapped_cossim(origin_id: int, key: str, ID_MAP) -> List[str]:
    mapped_ids = ID_MAP[str(origin_id)][f"{key}_top3_cossim"]
    return mapped_ids

def get_available_cache_dir():
    preferred = Path("/home/david/.cache")
    fallback = Path("/home/plowcow/.cache")

    if preferred.exists() and os.access(preferred, os.W_OK):
        return str(preferred)
    else:
        return str(fallback)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_forgotten_info(questions: List[str]) -> str:
    """1, 2, 3 … 형태로 개행 구분 포매팅."""
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

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
def build_commonsense_prompt(question: str, tokenizer, model_name, choices: List[Tuple[str, str]]) -> str:
    choice_block = "\n".join([f"{label}. {text}" for label, text in choices])
    user_msg = (
        f"{question}\n\nChoices:\n{choice_block}\n\n"
        "Include both the letter and the full correct answer."
    )

    prompt = wrap_prompt(user_msg, model_name)
    return prompt


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

# def load_model(base: str, lora: str, ds_cfg: str, dtype=torch.float16):
def load_model(base: str, ds_cfg: str, dtype=torch.float16):
    cfg = transformers.AutoConfig.from_pretrained(base)
    cfg.tp_size = 1

    model = AutoModelForCausalLM.from_pretrained(
        base,
        config=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        cache_dir=get_available_cache_dir(),
    )

    engine = deepspeed.init_inference(
        model,
        dtype=dtype,
        kernel_inject=False,
        replace_method="auto",
        config=json.load(open(ds_cfg)),
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

# ─────────────────────────────────────────────────────────────────────────────
# Batched generation (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def batched_generate(model, tok, prompts: List[str]) -> List[str]:
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_length = 256,
            do_sample=False,
            min_new_tokens=4,
            eos_token_id=tok.eos_token_id,
            use_cache=False,
        )

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

# ─────────────────────────────────────────────────────────────────────────────
# CommonsenseQA evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_commonsenseqa(truthfulqa, ID_MAP, split: str = "validation", batch_size: int = 4):
    ds = load_dataset("tau/commonsense_qa", split=split)

    id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in truthfulqa}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x  )

    preds, gold_labels, gold_texts, questions = [], [], [], []
    rouge_recalls = []                       # ➊ 신규 리스트
    par_negatives = 0
    par_positives = 0

    for batch in tqdm.tqdm(dl, desc=f"Evaluating {split}"):
        prompts = []
        batch_labels = []
        batch_gold_texts = []
        batch_questions = []
        preds_1 = []

        for ex in batch:
            labels = ex["choices"]["label"]        # ["A", "B", ...]
            texts  = ex["choices"]["text"]         # ["sand", ...]
            choices = list(zip(labels, texts))
            ref_q = mapped_question(ex["id"], "truthfulQA", id2question, ID_MAP)
            cos_sim = mapped_cossim(ex["id"], "truthfulQA", ID_MAP)
            max_cos_sim = max(float(x) for x in cos_sim) if cos_sim else 0.0

            if max_cos_sim > 0.8: # 0.8
                match = True
            else:
                match = False

            if not match:
                preds_1.append(0)
                par_negatives += 1
            else:
                preds_1.append(1)
                par_positives += 1

            batch_labels.append(ex["answerKey"])
            batch_questions.append(ex["question"].strip())

            # map answerKey -> full text for qualitative logs
            gold_text = dict(zip(labels, texts))[ex["answerKey"]]
            batch_gold_texts.append(gold_text)

        gold_labels.extend(batch_labels)
        gold_texts.extend(batch_gold_texts)
        questions.extend(batch_questions)

    print(f"\nPositives: {par_positives}, Negatives: {par_negatives}")

# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate model on CommonsenseQA")
    ap.add_argument("--ds_config", default="ds_config.json")
    ap.add_argument("--output_dir", default="./eval_results_commonsense")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--local_rank", type=int, default=-1)
    ap.add_argument("--id_mapping_json", default="csqa_to_truthqa_top3_ID_all.json", )
    ap.add_argument("--truthfulqa_json", default="../truthfulQA/truthfulQA_all_augmented_ID.json")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global MAPPING_PATH, ID_MAP

    with open(args.truthfulqa_json, encoding="utf-8") as f:
        data = json.load(f)

    # Load the split IDs
    with open("../truthfulQA/truthfulQA_continual_setting/TruthfulQA_split_ids.json", encoding="utf-8") as f:
        split_ids = json.load(f)

    ablation = 1  # 0, 1, 2, 3, 4, 5, 6
    stage = 123 # 1, 12, 123

    ablation_files = [
        "NQ_CURE_12K_a",
        "NQ_CURE_18K_a",
        "NQ_CURE_18K_a_no_b",
        "NQ_CURE_NO_HN_18K_a",
        "NQ_CURE_NO_HN_18K_a_no_b",
        "TQ_CURE_18K_a",
        "no_finetuning"
    ]

    # Convert the list to a set for fast lookup
    stage1_ids = set(split_ids["stage1"])
    stage1_stage2_ids = set(split_ids["stage1"]) | set(split_ids["stage2"])
    stage1_stage2_stage3_ids = (set(split_ids["stage1"]) | set(split_ids["stage2"]) | set(split_ids["stage3"]))

    if stage == 1:
        combined_ids = stage1_ids
        MAPPING_PATH = Path(f"../truthfulQA/truthfulQA_continual_setting/csqa_to_truthqa_top3_stage1_{ablation_files[ablation]}.json")
    elif stage == 12:
        combined_ids = stage1_stage2_ids
        MAPPING_PATH = Path(f"../truthfulQA/truthfulQA_continual_setting/csqa_to_truthqa_top3_stage1_2_{ablation_files[ablation]}.json")
    elif stage == 123:
        combined_ids = stage1_stage2_stage3_ids
        MAPPING_PATH = Path(f"../truthfulQA/truthfulQA_continual_setting/csqa_to_truthqa_top3_stage1_2_3_{ablation_files[ablation]}.json")

    # Filter data to include only examples with IDs in stage1
    filtered_data = [example for example in data if example["id"] in combined_ids]
    # print("len filtered data: ", len(filtered_data))

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        # ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)
        ID_MAP = json.load(f)

    # Evaluate
    eval_commonsenseqa(
        filtered_data, ID_MAP, split=args.split, batch_size=args.batch_size
    )

    ### Save
    # out_samples = Path(args.output_dir) / "commonsenseqa_samples.json"
    # out_agg = Path(args.output_dir) / "commonsenseqa_metrics.json"
    # out_samples.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    # out_agg.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    #
    # print("\n✅ Saved per‑sample results to", out_samples)
    # print("✅ Saved aggregate metrics to", out_agg)
    # print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
