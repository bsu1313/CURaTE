import os, json, argparse, re, tqdm
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed, transformers
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import random
import sys


lora_path_name = ""

REFUSAL_PATH = Path("../refusal_answer.json")  
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))


def get_available_cache_dir():
    preferred = Path("/home/.cache")
    fallback = Path("/home/plowcow/.cache")

    if preferred.exists() and os.access(preferred, os.W_OK):
        return str(preferred)
    else:
        return str(fallback)


def mapped_question(origin_id: int, key: str, id2question, ID_MAP) -> List[str]:
    try:
        mapped_ids = ID_MAP[str(origin_id)][f"{key}_top3_ids"]
        return [id2question[mid] for mid in mapped_ids if mid in id2question]
    except (KeyError, IndexError):
        return id2question[origin_id]


def mapped_cossim(origin_id: int, key: str, ID_MAP) -> List[str]:
    mapped_ids = ID_MAP[str(origin_id)][f"{key}_top3_cossim"]
    return mapped_ids


def format_forgotten_info(questions: List[str]) -> str:
    return "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])


class QADataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def wrap_prompt(p, if_llama):
    # if 'llama-3' in if_llama or 'llama_3' in if_llama:
    #     question_start_token = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Jul 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    #     question_end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # elif 'llama-2' in if_llama or 'llama_2' in if_llama:
    #     question_start_token = "[INST] "
    #     question_end_token = " [/INST]"
    # else:
    #     raise ValueError('Please provide llama model')
    return p
    # return f"{question_start_token}{p}{question_end_token}"



def _mean(x: List[float]) -> float:
    return float(np.mean(x)) if x else 0.0


def acc_contains(pred: str, truth: str) -> int:
    return int(bool(re.search(re.escape(truth), pred, re.I)))


def postprocess_completion(comp: str) -> str:
    cut = comp.find("[Reason]")
    if cut != -1:
        comp = comp[:cut]
    return comp.strip()


def eval_subset(name, data: List[Dict[str, Any]], forget_data, ID_MAP, batch_size: int = 4):
    def identity_collate(batch):
        return batch

    if name == "obqa" or name == "csqa":
        question = "instruction"
    else:
        question = "question"
    if name == "NU":
        ans = "contrastive_answer"
    else:
        ans = "answer"

    id2question: dict[int, str] = {ex["id"]: ex["instruction"] for ex in forget_data}

    dl = DataLoader(QADataset(data), batch_size=batch_size, collate_fn=identity_collate)
    all_results = []
    par_positives = 0
    par_negatives = 0

    if name == "forget":
        input_question = "paraphrased_instruction_input"
        search_question = "paraphrased_instruction"
    elif name == "NU":
        input_question = "contrastive_instruction_input"
        search_question = "contrastive_instruction"
    elif name == "retain" or name == "obqa" or name == "csqa":
        input_question = "text_input"
        search_question = "instruction"

    for batch in tqdm.tqdm(dl, desc=f"Evaluating subset {name}"):
        prompts_1, ids_1, q1_inputs, preds_1, answers_1 = [], [], [], [], []

        for item in batch:
            # Case 1: paraphrased question
            if item.get(input_question):
                ref_q = mapped_question(item["id"], "forget_data", id2question, ID_MAP)
                cos_sim = mapped_cossim(item["id"], "forget_data", ID_MAP)
                max_cos_sim = max(float(x) for x in cos_sim) if cos_sim else 0.0

                if max_cos_sim > 0.9: # 0.9
                    match = True
                else:
                    match = False

                if not match:
                    preds_1.append(0)
                    par_negatives += 1
                else:
                    preds_1.append(1)
                    par_positives += 1

                answers_1.append(item[ans])
                ids_1.append(item["id"])
                q1_inputs.append({
                    "id": item["id"],
                    "forgotten_info": item[question],
                    "query": item[input_question]
                })

    print("Results for subset:", name)
    print(f"\nParaphrased positives: {par_positives}, negatives: {par_negatives}")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_config", default="ds_config.json")
    ap.add_argument("--output_dir", default="./eval_results")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--local_rank", type=int, default=-1)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stages = {}

    ablation = 1  # 0, 1, 2, 3, 4, 5, 6
    stage = 4 # 1, 2, 3, 4

    ablation_files = [
        "NQ_CURaTE_12K_a",
        "NQ_CURaTE_18K_a",
        "NQ_CURaTE_18K_a_no_b",
        "NQ_CURaTE_NO_HN_18K_a",
        "NQ_CURaTE_NO_HN_18K_a_no_b",
        "TQ_CURaTE_18K_a",
        "no_finetuning"
    ]

    if stage == 1:
        MAPPING_PATH = Path(f"./ScienceQA_to_stage1_top3_{ablation_files[ablation]}.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_train_SD.json"), encoding="utf-8") as f:
            forget_data = json.load(f)
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_train_SD.json"), encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_test_RD.json"), encoding="utf-8") as f:
            stages["retain"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_train_SD.json"), encoding="utf-8") as f:
            stages["NU"] = json.load(f)
    elif stage == 2:
        MAPPING_PATH = Path(f"./ScienceQA_to_stage2_top3_{ablation_files[ablation]}.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_train_SD.json"), encoding="utf-8") as f:
            forget_data = json.load(f)
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_train_SD.json"), encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_physics_test_RD.json"), encoding="utf-8") as f:
            stages["retain"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_physics_train_SD.json"), encoding="utf-8") as f:
            stages["NU"] = json.load(f)
    elif stage == 3:
        MAPPING_PATH = Path(f"./ScienceQA_to_stage3_top3_{ablation_files[ablation]}.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_chemistry_train_SD.json"), encoding="utf-8") as f:
            forget_data = json.load(f)
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_chemistry_train_SD.json"), encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_physics_chemistry_test_RD.json"), encoding="utf-8") as f:
            stages["retain"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_physics_chemistry_train_SD.json"), encoding="utf-8") as f:
            stages["NU"] = json.load(f)
    elif stage == 4:
        MAPPING_PATH = Path(f"./ScienceQA_to_stage4_top3_{ablation_files[ablation]}.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_chemistry_economics_train_SD.json"), encoding="utf-8") as f:
            forget_data = json.load(f)
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_chemistry_economics_train_SD.json"), encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_physics_chemistry_economics_test_RD.json"), encoding="utf-8") as f:
            stages["retain"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_physics_chemistry_economics_train_SD.json"), encoding="utf-8") as f:
            stages["NU"] = json.load(f)

    with open(os.path.join("test_utility", f"processed_openbookqa_test.json"), encoding="utf-8") as f:
        stages["obqa"] = json.load(f)
    with open(os.path.join("test_utility", f"processed_commonqa_test.json"), encoding="utf-8") as f:
        stages["csqa"] = json.load(f)

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)

    # Evaluate
    total_results: Dict[str, Dict] = {}
    output_data = {}
    for name, ds in stages.items():
        eval_subset(name, ds, forget_data, ID_MAP, batch_size=args.batch_size, )
        # total_results[name] = results

        # grouped = {"paraphrased": []}
        # for r in results:
        #     if r["type"] in grouped:
        #         grouped[r["type"]].append(r["acc_score"])
        #
        # aggregate_scores = {
        #     k: float(np.mean(v)) if v else 0.0
        #     for k, v in grouped.items()
        # }
        # print(f"[{name}] Aggregate scores: {json.dumps(aggregate_scores, indent=2)}")
        # # print(f"[{name}] {json.dumps(results, indent=2, ensure_ascii=False)}")
        # output_data[name] = {
        #     "metrics": aggregate_scores,
        #     "samples": results
        # }

    # # Save
    # out_path = os.path.join(args.output_dir, "scienceQA_results.json")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(total_results, f, indent=2, ensure_ascii=False)
    #
    # print(f"\n✅ Saved evaluation to {out_path}")
    #
    # out_path = os.path.join(args.output_dir, "scienceQA_results_summary.json")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(output_data, f, indent=2, ensure_ascii=False)
    #
    # print(f"\n✅ Saved evaluation with metrics to {out_path}")
    # # print(json.dumps(aggregate_scores, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
