import os, json, argparse, re, tqdm
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed, transformers
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from pathlib import Path
import random
import sys

# Prompt template helpers

lora_path_name = ""

# # MAPPING_PATH = Path("./truthfulQA_top3_id_mappings_all.json")  # ← vanilla sent emb model
# MAPPING_PATH = Path("./top3_id_mappings_all.json")  # ← finetuned sent emb model
# with MAPPING_PATH.open("r", encoding="utf-8") as f:
#     ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)

REFUSAL_PATH = Path("../refusal_answer.json")  # ← 실제 파일명/경로
REF_PHRASES: list[str] = json.loads(REFUSAL_PATH.read_text(encoding="utf-8"))


def get_available_cache_dir():
    preferred = Path("/home/david/.cache")
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


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────
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


def build_llama2_prompt(question: str, forgotten_info: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return plain text prompt, not token IDs
        add_generation_prompt=True  # Adds the assistant's turn prefix
    )
    return prompt


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
        cache_dir=get_available_cache_dir(),
    )
    # model = PeftModel.from_pretrained(model, lora).merge_and_unload()

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
        cache_dir=get_available_cache_dir(),
    )
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return engine.module, tok


# ─────────────────────────────────────────────────────────────────────────────
# Batched generation
# ─────────────────────────────────────────────────────────────────────────────

def batched_generate(model, tok, prompts: List[str]) -> List[str]:
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    # print("PROMPTS : ", prompts)

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            min_new_tokens=4,
            eos_token_id=tok.eos_token_id,
            use_cache=False,
        )

    results = []
    for prompt, generated_ids in zip(prompts, outs):
        full_text = tok.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()
        # print("FULL TEXT : ", full_text)

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
        # print("ANSWER : ", answer)
        results.append(answer)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Custom tofu evaluation logic
# ─────────────────────────────────────────────────────────────────────────────

def eval_subset(model, tok, model_name, name, data: List[Dict[str, Any]], sent_model, ID_MAP, batch_size: int = 4):
    def identity_collate(batch):
        return batch

    # print("data sample: ", data[0])
    if name == "obqa" or name == "csqa":
        question = "instruction"
    else:
        question = "question"

    id2question: dict[int, str] = {ex["id"]: ex[question] for ex in data}

    dl = DataLoader(QADataset(data), batch_size=batch_size, collate_fn=identity_collate)
    all_results = []
    par_positives = 0
    par_negatives = 0

    if name == "forget":
        input_question = "paraphrased_instruction_input"
    elif name == "NU":
        input_question = "contrastive_instruction_input"
    elif name == "retain" or name == "obqa" or name == "csqa":
        input_question = "text_input"

    for batch in tqdm.tqdm(dl, desc=f"Evaluating subset {name}"):
        # prompts_1, refs_1, ids_1, q1_inputs, preds_1, incorrect_1 = [], [], [], [], [], []
        prompts_1, ids_1, q1_inputs, preds_1, answers_1 = [], [], [], [], []

        for item in batch:
            # print("item: ", item)

            # Case 1: paraphrased question
            if item.get(input_question):
                ref_q = mapped_question(item["id"], "forget_data", id2question, ID_MAP)
                cos_sim = mapped_cossim(item["id"], "forget_data", ID_MAP)
                max_cos_sim = max(float(x) for x in cos_sim) if cos_sim else 0.0

                # if max_cos_sim > 0.8:
                #     match = True
                # else:
                #     match = False
                match = False

                # match = False
                # for f_info in ref_q:
                #     q_emb = sent_model.encode(item["paraphrased_question"], convert_to_tensor=True)
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

                ref_q = format_forgotten_info(ref_q)

                # prompts_1.append(build_llama2_prompt(item["paraphrased_question"], ref_q, tok))
                prompts_1.append(wrap_prompt(item[input_question], model_name.lower()))
                answers_1.append(item["answer"])

                # refs_1.append(item["prediction"])
                ids_1.append(item["id"])
                q1_inputs.append({
                    "id": item["id"],
                    "forgotten_info": item[question],
                    "query": item[input_question]
                })

        # print("prompts_1: ", prompts_1)
        # print("answers_1: ", answers_1)
        # Generate responses
        if prompts_1:
            gens_1 = batched_generate(model, tok, prompts_1)
            # print("gens_1: ", gens_1)

            for i, pred in enumerate(preds_1):
                if pred == 1:
                    gens_1[i] = random.choice(REF_PHRASES)
                elif pred == 0:
                    gens_1[i] = gens_1[i].strip()
                else:
                    raise ValueError(f"Unexpected prediction class: {pred}")

            correct = 0
            results = []
            outputs = []
            gt = []
            pattern = re.compile(r'The answer is ([A-Z]).')
            res = [pattern.findall(otp) for otp in gens_1]
            # print("res: ", res)
            pred = []
            for r_i in range(len(res)):
                if len(res[r_i]) == 1:
                    answer = res[r_i][0]  # 'A', 'B', ...
                else:
                    answer = "FAILED"
                #     print("*******************************************", res[r_i])
                pred.append(answer)
                results.append(res[r_i])
                # outputs.append(output[r_i])
                # gt.append(answers[r_i])

                if str(answer) == str(answers_1[r_i]):
                    correct += 1
                    acc_score = 1
                    # print('correct:', str(answer), str(answers_1[r_i]))
                else:
                    acc_score = 0
                    # print('gt-ans:', str(answer), str(answers_1[r_i]))

                all_results.append({
                    "id": ids_1[i],
                    "type": "paraphrased",
                    "input": prompts_1[i],
                    "generated": gens_1[i],
                    # "reference": refs_1[i],
                    "acc_score": acc_score
                })

            acc = correct / len(results) * 100
            # print(f"Accuracy: {acc:.2f}%")

    print(f"\nParaphrased positives: {par_positives}, negatives: {par_negatives}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="gcyzsl/O3_LLAMA2_ScienceQA")
    # ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--ds_config", default="ds_config.json")
    ap.add_argument("--output_dir", default="./eval_results")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--local_rank", type=int, default=-1)

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tok = load_model(args.base_model, args.ds_config)

    sent_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    # model_dir = "../mpnet_contrastive_model"
    # sent_model = SentenceTransformer(model_dir)

    stages = {}
    stage = 1

    if stage == 1:
        MAPPING_PATH = Path("./ScienceQA_to_stage1_top3.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_train_SD.json"), encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_train_SD.json"), encoding="utf-8") as f:
            stages["NU"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_test_RD.json"), encoding="utf-8") as f:
            stages["retain"] = json.load(f)
    elif stage == 2:
        MAPPING_PATH = Path("./ScienceQA_to_stage2_top3.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_train_SD.json"), encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_physics_train_SD.json"), encoding="utf-8") as f:
            stages["NU"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_physics_test_RD.json"),
                  encoding="utf-8") as f:
            stages["retain"] = json.load(f)
    elif stage == 3:
        MAPPING_PATH = Path("./ScienceQA_to_stage3_top3.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_chemistry_train_SD.json"),
                  encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_physics_chemistry_train_SD.json"),
                  encoding="utf-8") as f:
            stages["NU"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_physics_chemistry_test_RD.json"),
                  encoding="utf-8") as f:
            stages["retain"] = json.load(f)
    elif stage == 4:
        MAPPING_PATH = Path("./ScienceQA_to_stage4_top3.json")
        with open(os.path.join("test_forget_PR", f"PR_scienceqa_biology_physics_chemistry_economics_train_SD.json"),
                  encoding="utf-8") as f:
            stages["forget"] = json.load(f)
        with open(os.path.join("test_NU", f"NU_scienceqa_biology_physics_chemistry_economics_train_SD.json"),
                  encoding="utf-8") as f:
            stages["NU"] = json.load(f)
        with open(os.path.join("retain", f"processed_scienceqa_not_biology_physics_chemistry_economics_test_RD.json"),
                  encoding="utf-8") as f:
            stages["retain"] = json.load(f)

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
        results = eval_subset(model, tok, args.base_model, name, ds, sent_model, ID_MAP, batch_size=args.batch_size, )
        total_results[name] = results

        grouped = {"paraphrased": []}
        for r in results:
            if r["type"] in grouped:
                grouped[r["type"]].append(r["acc_score"])

        aggregate_scores = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in grouped.items()
        }
        print(f"[{name}] Aggregate scores: {json.dumps(aggregate_scores, indent=2)}")
        # print(f"[{name}] {json.dumps(results, indent=2, ensure_ascii=False)}")
        output_data[name] = {
            "metrics": aggregate_scores,
            "samples": results
        }

    # Save
    out_path = os.path.join(args.output_dir, "scienceQA_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(total_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved evaluation to {out_path}")

    out_path = os.path.join(args.output_dir, "scienceQA_results_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved evaluation with metrics to {out_path}")
    # print(json.dumps(aggregate_scores, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
