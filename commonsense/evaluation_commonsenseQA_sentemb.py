# -*- coding: utf-8 -*-


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

from datasets import load_dataset  





REFUSAL_PATH = Path("../refusal_answer.json")   
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))

def mapped_question(origin_id: int, key: str, id2question, ID_MAP) -> List[str]:

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


def format_forgotten_info(questions: List[str]) -> str:
    
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
    
    m = re.search(r"[A-E]", pred)
    return m.group(0) if m else None

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



def batched_generate(model, tok, prompts: List[str]) -> List[str]:
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    # print("prompts:", prompts)

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            # max_new_tokens=128,
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


        if full_text.startswith(prompt_text):
            answer = full_text[len(prompt_text):].strip()
        else:
            idx = full_text.find(prompt_text)
            if idx != -1:
                answer = full_text[idx + len(prompt_text):].strip()
            else:
                answer = full_text
        results.append(answer)
    return results

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

def eval_commonsenseqa(model, tok, model_name, truthfulqa, sent_model, ID_MAP, split: str = "validation", batch_size: int = 4):
    ds = load_dataset("tau/commonsense_qa", split=split)


    id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in truthfulqa}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x  )

    preds, gold_labels, gold_texts, questions = [], [], [], []
    rouge_recalls = []                     
    par_negatives = 0
    par_positives = 0

    for batch in tqdm.tqdm(dl, desc=f"Evaluating {split}"):
        prompts = []
        batch_labels = []
        batch_gold_texts = []
        batch_questions = []
        preds_1 = []

        for ex in batch:
            labels = ex["choices"]["label"]        
            texts  = ex["choices"]["text"]         
            choices = list(zip(labels, texts))
            ref_q = mapped_question(ex["id"], "truthfulQA", id2question, ID_MAP)
            cos_sim = mapped_cossim(ex["id"], "truthfulQA", ID_MAP)
            max_cos_sim = max(float(x) for x in cos_sim) if cos_sim else 0.0

            if max_cos_sim > 0.8:
                match = True
            else:
                match = False

            # match = False
            # q_emb = sent_model.encode(ex["question"], convert_to_tensor=True)
            # for f_info in ref_q:
            #     f_emb = sent_model.encode(f_info, convert_to_tensor=True)
            #     cos_sim = util.cos_sim(q_emb, f_emb)
            #     if cos_sim.item() > 0.8:  # threshold for similarity
            #         match = True
            if not match:
                preds_1.append(0)
                par_negatives += 1
            else:
                preds_1.append(1)
                par_positives += 1
            
            # n_forgotten =3
            # if ID_MAP is not None and n_forgotten > 0:
            #     if ex["question"] not in ID_MAP:
            #         raise KeyError(
            #             f"[ID_MAP]에 해당 질문이 없습니다: «{ex['question']}»"
            #         )
            #     cand_list = ID_MAP[ex["question"]][:n_forgotten]
            # else:
            #     cand_list = []
            # # print("ID_MAP:", ID_MAP)
            # # print("ex['question']:", ex["question"])
            # # print('cand_list:', cand_list)
            # if len(cand_list) > 0:
            #     match = False
            #     for f_info in cand_list:
            #         q_emb = sent_model.encode(ex["question"], convert_to_tensor=True)
            #         f_emb = sent_model.encode(f_info, convert_to_tensor=True)
            #         cos_sim = util.cos_sim(q_emb, f_emb)
            #         if cos_sim.item() > 0.8:  # threshold for similarity
            #             match = True
            #     if not match:
            #         preds_1.append(0)
            #         par_negatives += 1
            #     else:
            #         preds_1.append(1)
            #         par_positives += 1

            if len(ref_q) == 1:
                forgotten_info = ref_q[0]          
            else:
                forgotten_info = format_forgotten_info(ref_q) if ref_q else ""
            

            prompts.append(build_commonsense_prompt(ex["question"], tok, model_name.lower(), choices))
            batch_labels.append(ex["answerKey"])
            batch_questions.append(ex["question"].strip())

   
            gold_text = dict(zip(labels, texts))[ex["answerKey"]]
            batch_gold_texts.append(gold_text)


        gens = batched_generate(model, tok, prompts)

        for i, pred in enumerate(preds_1):
            if pred == 1:
                gens[i] = random.choice(REF_PHRASES)
            elif pred == 0:
                gens[i] = gens[i].strip()
            else:
                raise ValueError(f"Unexpected prediction class: {pred}")
    
    
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
    print(f"\nPositives: {par_positives}, Negatives: {par_negatives}")
    return aggregate, samples


def main():
    ap = argparse.ArgumentParser(description="Evaluate model on CommonsenseQA")

    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
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

    model, tok = load_model(args.base_model, args.ds_config)

    model_dir = "../mpnet_contrastive_model"
    sent_model = SentenceTransformer(model_dir)

    with open(args.truthfulqa_json, encoding="utf-8") as f:
        data = json.load(f)


    with open("../truthfulQA/truthfulQA_continual_setting/TruthfulQA_split_ids.json", encoding="utf-8") as f:
        split_ids = json.load(f)

    stage = 12

    stage1_ids = set(split_ids["stage1"])
    stage1_stage2_ids = set(split_ids["stage1"]) | set(split_ids["stage2"])
    stage1_stage2_stage3_ids = (set(split_ids["stage1"]) | set(split_ids["stage2"]) | set(split_ids["stage3"]))

    if stage == 1:
        combined_ids = stage1_ids
        MAPPING_PATH = Path("../truthfulQA/truthfulQA_continual_setting/csqa_to_truthqa_top3_stage1.json")
    elif stage == 12:
        combined_ids = stage1_stage2_ids
        MAPPING_PATH = Path("../truthfulQA/truthfulQA_continual_setting/csqa_to_truthqa_top3_stage1_2.json")
    elif stage == 123:
        combined_ids = stage1_stage2_stage3_ids
        MAPPING_PATH = Path("../truthfulQA/truthfulQA_continual_setting/csqa_to_truthqa_top3_stage1_2_3.json")

    # Filter data to include only examples with IDs in stage1
    filtered_data = [example for example in data if example["id"] in combined_ids]
    # print("len filtered data: ", len(filtered_data))

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        # ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)
        ID_MAP = json.load(f)

    # Evaluate
    aggregate, samples = eval_commonsenseqa(
        model, tok, args.base_model, filtered_data, sent_model, ID_MAP, split=args.split, batch_size=args.batch_size
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
