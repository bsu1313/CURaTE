import os, sys, json, math, argparse, re, random, tqdm
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import deepspeed, transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from torch.utils.data.dataloader import default_collate
from pathlib import Path
import sys

# Prompt template helpers
from conversation import get_conv_template  # ensure import path is correct

lora_path_name = ""


# MAPPING_PATH = Path("./truthfulQA_top3_id_mappings_all.json")  # ← vanilla sent emb model
MAPPING_PATH = Path("./top3_id_mappings_all.json")  # ← finetuned sent emb model
with MAPPING_PATH.open("r", encoding="utf-8") as f:
    ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)
    
REFUSAL_PATH = Path("./truthfulQA_refusal_answer.json")   # ← 실제 파일명/경로
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))

def get_available_cache_dir():
    preferred = Path("/home/david/.cache")
    fallback = Path("/home/plowcow/.cache")

    if preferred.exists() and os.access(preferred, os.W_OK):
        return str(preferred)
    else:
        return str(fallback)

def mapped_question(origin_id: int, key: str, id2question) -> List[str]:
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
    
def format_forgotten_info(questions: List[str]) -> str:
    return "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

def get_related_ids(example_id: int) -> tuple[int, int]:
    entry = ID_MAP[str(example_id)]
    return entry["paraphrased_top3_ids"][0], entry["contrastive_top3_ids"][0]


class QADataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
def build_llama2_prompt(question: str, forgotten_info: str) -> str:
    conv = get_conv_template("llama-2")
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()
    # final_prompt = "[INST] " + question + " [\INST]"
    return final_prompt
# def build_llama2_prompt(question: str, forgotten_info: str) -> str:
#     if "newinst2" in lora_path_name:
#         input_text = (
#         "Based on the [Forgotten Information], decide whether to answer or "
#         "refuse to answer the [Query]. Then provide an appropriate response "
#         "accordingly. If you choose to refuse, include the corresponding "
#         "[Reason] for your decision.\n\n"
#         f"[Forgotten Information]:\n{forgotten_info}\n\n"
#         f"[Query]:\n{question}"
#     )
#     elif "newinst3_1HN" in lora_path_name:
#         input_text = f"Given: [Forgotten Information]: 1. ... 2. ... (and possibly more) [Query]: ... If the [Query] is related to any of the listed items, refuse to answer and explain which item it relates to, along with a [Reason]. If unrelated to all, answer the query appropriately.\n\n[Forgotten Information]:\n{forgotten_info}[Query]:\n{question}"
#     elif "newinst3_2HN" in lora_path_name:  # no reason
#         input_text = f"Given: [Forgotten Information]: 1. ... 2. ... (and possibly more) [Query]: ... \n\n If the [Query] is related to any of the listed question or sentence, refuse to answer. If unrelated to all, answer the query appropriately.\n\n[Forgotten Information]:\n{forgotten_info}[Query]:\n{question}"
#
#     elif "newinst4" in lora_path_name:
#         input_text = f"Given: [Forgotten Information]: 1. ... 2. ... (and possibly more) [Query]: ... \n\n If the [Query] is related to any of the listed question or sentence, refuse to answer and explain along with a [Reason]. If unrelated to all, answer the query appropriately.\n\n[Forgotten Information]:\n{forgotten_info}[Query]:\n{question}"
#
#     elif "NQ_LTU" in lora_path_name:
#         input_text = f"Based on the [Forgotten Information], decide whether to answer or refuse to answer the [Query]. Then provide an appropriate response accordingly. If you choose to refuse, include the corresponding [Reason] for your decision.\n\n[Forgotten Information]:\n{forgotten_info}\n\n[Query]:\n{question}"
#
#     # elif "newinst5" in lora_path_name:
#     #     input_text = f"Based on the [Forgotten Information], decide whether to answer or refuse to answer the [Query]. Then provide an appropriate response accordingly.\n\n[Forgotten Information]:\n{forgotten_info}[Query]:\n{question}"
#
#
#     # elif "newinst" in lora_path_name:
#     #     input_text = f"Based on the [Forgotten Information], decide whether to answer or refuse to answer the [Query]. Then provide an appropriate response accordingly.\n\n[Forgotten Information]:\n{forgotten_info}[Query]:\n{question}"
#
#
#
#
#     # elif lora_path_name == "/mnt/hdd/seyun_workspace/cache_LTE/output_lte_lora_llama-2_7b_chat_tofu_filted_B_and_new_forget_add_to_A_reason_more_lora_weight":
#     #     input_text = f"Please read the information below, but do not respond to it or provide an answer to the following query.\n\n[Forgotten Information]:\n{forgotten_info}\n\n[Query]:\n{question}"
#     else:
#         input_text = f"[Forgotten Information]:\n{forgotten_info}\n\n[Query]:\n{question}"
#         raise ValueError
#
#     conv = get_conv_template("llama-2")
#     conv.set_system_message("You are a helpful, respectful and honest assistant.")
#     conv.append_message(conv.roles[0], input_text)
#     conv.append_message(conv.roles[1], None)
#     return conv.get_prompt()

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def _mean(x: List[float]) -> float:
    return float(np.mean(x)) if x else 0.0

def acc_contains(pred: str, truth: str) -> int:
    return int(bool(re.search(re.escape(truth), pred, re.I)))

def postprocess_completion(comp: str) -> str:
    cut = comp.find("[Reason]")
    if cut != -1:
        comp = comp[:cut]
    return comp.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
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
        cache_dir= get_available_cache_dir(),
    )
    # model = PeftModel.from_pretrained(model, lora).merge_and_unload()

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
        cache_dir= get_available_cache_dir(),
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return engine.module, tok

# ─────────────────────────────────────────────────────────────────────────────
# Batched generation
# ─────────────────────────────────────────────────────────────────────────────

def batched_generate(model, tok, prompts: List[str]) -> List[str]:
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            min_new_tokens=4,
            eos_token_id=tok.eos_token_id,
            use_cache=False,
        )
        # outs = model.generate(**inputs,
        #                       # max_new_tokens=256,
        #                       max_length = 200,
        #                       do_sample=False,
        #                       # min_new_tokens=4,
        #                       eos_token_id=tok.eos_token_id,
        #                       use_cache=False)

    results = []
    for ids in outs:
        full = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        comp = full.split("[/INST]", 1)[-1].strip()
        results.append(postprocess_completion(comp))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Custom tofu evaluation logic
# ─────────────────────────────────────────────────────────────────────────────
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

def eval_tofu_custom(model, tok, data: List[Dict[str, Any]], sent_model, batch_size: int = 4):
    
    def identity_collate(batch):
        return batch
    
    # print("data sample: ", data[0])
    id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in data}
    
    dl = DataLoader(QADataset(data), batch_size=batch_size, collate_fn=identity_collate)
    all_results = []
    par_positives = 0
    par_negatives = 0
    con_positives = 0
    con_negatives = 0

    for batch in tqdm.tqdm(dl, desc="Evaluating custom tofu"):
        # prompts_1, refs_1, ids_1, q1_inputs, preds_1, incorrect_1 = [], [], [], [], [], []
        prompts_1, ids_1, q1_inputs, preds_1, incorrect_1 = [], [], [], [], []
        prompts_2, refs_2, ids_2, q2_inputs, preds_2 = [], [], [], [], []

        for item in batch:
            # print("item: ", item)
            
            # Case 1: paraphrased question
            if item.get("paraphrased_question"):
                ref_q = mapped_question(item["id"], "paraphrased", id2question)

                # roberta_prompts = ["[Forgotten Information]:\n" + f_info + "\n\n[Query]:\n" + item["paraphrased_question"]
                #     for f_info in ref_q
                # ]
                # predictions = predict(roberta_prompts, roberta_tok, roberta_model)
                # preds = [p["pred_class"] for p in predictions]

                match = False
                for f_info in ref_q:
                    q_emb = sent_model.encode(item["paraphrased_question"], convert_to_tensor=True)
                    f_emb = sent_model.encode(f_info, convert_to_tensor=True)
                    cos_sim = util.cos_sim(q_emb, f_emb)
                    if cos_sim.item() > 0.8:  # threshold for similarity
                        match = True

                if not match:
                    preds_1.append(0)
                    par_negatives += 1
                else:
                    preds_1.append(1)
                    par_positives += 1

                ref_q = format_forgotten_info(ref_q)
                
                prompts_1.append(build_llama2_prompt(item["paraphrased_question"], ref_q))
                incorrect_1.append([s.strip() for s in item["Incorrect Answers"].split(";")])

                # refs_1.append(item["prediction"])
                ids_1.append(item["id"])
                q1_inputs.append({
                    "id": item["id"],
                    "forgotten_info": item["question"],
                    "query": item["paraphrased_question"]
                })

            # Case 2: contrastive question
            if item.get("contrastive_question") and item.get("contrastive_answer"):
                ref_q = mapped_question(item["id"], "contrastive", id2question)

                # roberta_prompts = ["[Forgotten Information]:\n" + f_info + "\n\n[Query]:\n" + item["contrastive_question"]
                #     for f_info in ref_q
                # ]
                # predictions = predict(roberta_prompts, roberta_tok, roberta_model)
                # preds = [p["pred_class"] for p in predictions]


                match = False
                for f_info in ref_q:
                    q_emb = sent_model.encode(item["contrastive_question"], convert_to_tensor=True)
                    f_emb = sent_model.encode(f_info, convert_to_tensor=True)
                    cos_sim = util.cos_sim(q_emb, f_emb)
                    if cos_sim.item() > 0.8:  # threshold for similarity
                        match = True

                if not match:
                    preds_2.append(0)
                    con_negatives += 1
                else:
                    preds_2.append(1)
                    con_positives += 1
                
                ref_q = format_forgotten_info(ref_q)
                
                prompts_2.append(build_llama2_prompt(item["contrastive_question"], ref_q))
                # print("prompts_2: ", prompts_2)
                # print("preds_2: ", preds_2)
                refs_2.append(item["contrastive_answer"])
                ids_2.append(item["id"])
                q2_inputs.append({
                    "id": item["id"],
                    "forgotten_info": item["question"],
                    "query": item["contrastive_question"]
                })

        # Generate responses
        if prompts_1:
            gens_1 = batched_generate(model, tok, prompts_1)

            for i, pred in enumerate(preds_1):
                if pred == 1:
                    gens_1[i] = REF_PHRASES[0]
                elif pred == 0:
                    gens_1[i] = gens_1[i].strip()
                else:
                    raise ValueError(f"Unexpected prediction class: {pred}")

            for i in range(len(gens_1)):
                
                
                # rouge_score = max(
                #     rouge.score(ref, gens_1[i])["rougeL"].recall
                #     for ref in REF_PHRASES
                # )
                rouge_score = max(
                    rouge.score(ref, gens_1[i])["rougeL"].recall
                    for ref in incorrect_1[i]
                )
                                
                
                all_results.append({
                    "id": ids_1[i],
                    "type": "paraphrased",
                    "input": q1_inputs[i],
                    "generated": gens_1[i],
                    # "reference": refs_1[i],
                    "rougeL_recall": rouge_score
                })

        if prompts_2:
            gens_2 = batched_generate(model, tok, prompts_2)

            for i, pred in enumerate(preds_2):
                if pred == 1:
                    gens_2[i] = REF_PHRASES[0]
                elif pred == 0:
                    gens_2[i] = gens_2[i].strip()
                else:
                    raise ValueError(f"Unexpected prediction class: {pred}")

            for i in range(len(gens_2)):
                rouge_score = rouge.score(refs_2[i], gens_2[i])["rougeL"].recall
                all_results.append({
                    "id": ids_2[i],
                    "type": "contrastive",
                    "input": q2_inputs[i],
                    "generated": gens_2[i],
                    "reference": refs_2[i],
                    "rougeL_recall": rouge_score
                })

    print(f"\nParaphrased positives: {par_positives}, negatives: {par_negatives}")
    print(f"Contrastive positives: {con_positives}, negatives: {con_negatives}")
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--base_model", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-chat-hf")
    # ap.add_argument("--lora_path", required=True)
    # ap.add_argument("--ds_config", required=True)
    ap.add_argument("--ds_config", default="ds_config.json")
    ap.add_argument("--output_dir", default="./eval_results")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--local_rank", type=int, default=-1)
    # ap.add_argument("--custom_data_json", required=True)
    # ap.add_argument("--custom_data_json", default="./truthfuQA_consent_false_only_augmented_llama_gen_consent_true_only.json")
    # ap.add_argument("--custom_data_json", default="./truthfulQA_enriched.json")
    ap.add_argument("--custom_data_json", default="./truthfulQA_all_augmented_ID.json")
    
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # global lora_path_name
    # lora_path_name = args.lora_path

    # Load model
    # model, tok = load_model(args.base_model, args.lora_path, args.ds_config)
    model, tok = load_model(args.base_model, args.ds_config)

    # model_dir = "../roberta_features_Aprime_classifier"
    # roberta_tok = RobertaTokenizer.from_pretrained(model_dir)
    # roberta_model = RobertaForSequenceClassification.from_pretrained(model_dir)
    # roberta_model.eval()
    model_dir = "../mpnet_contrastive_model"
    sent_model = SentenceTransformer(model_dir)

    # Load new data
    with open(args.custom_data_json, encoding="utf-8") as f:
        data = json.load(f)

    # Evaluate
    results = eval_tofu_custom(model, tok, data, sent_model, batch_size=args.batch_size)

    # Save
    out_path = os.path.join(args.output_dir, "truthfulQA_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved evaluation to {out_path}")
    
    
    
    # 5) Calculate and log aggregate Rouge-L
    grouped = {"paraphrased": [], "contrastive": []}
    for r in results:
        if r["type"] in grouped:
            grouped[r["type"]].append(r["rougeL_recall"])

    aggregate_scores = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in grouped.items()
    }

    # 6) Save full results + aggregate
    output_data = {
        "metrics": aggregate_scores,
        "samples": results
    }

    out_path = os.path.join(args.output_dir, "truthfulQA_result_summery.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved evaluation with metrics to {out_path}")
    print(json.dumps(aggregate_scores, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
