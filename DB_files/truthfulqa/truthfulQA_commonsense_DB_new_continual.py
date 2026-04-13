"""
Build three CSQA-to-TruthfulQA top-3-mapping files,
constrained to (1) stage1-ids, (2) stage1+2-ids, (3) stage1+2+3-ids.
"""

import json, torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path


baseline_model = "mpnet" # mpnet, minilm, distilroberta
ablation = 1 # 0, 1, 2, 3, 4, 5, 6

ablation_files = [
    "NQ_CURaTE_12K_a",
    "NQ_CURaTE_18K_a",
    "NQ_CURaTE_18K_a_no_b",
    "NQ_CURaTE_NO_HN_18K_a",
    "NQ_CURaTE_NO_HN_18K_a_no_b",
    "TQ_CURaTE_18K_a",
    "no_finetuning"
]

truth_file  = "../../truthfulQA/truthfulQA_all_augmented_ID.json"
stage_file  = "../../truthfulQA/truthfulQA_continual_setting/TruthfulQA_split_ids.json"        # {"stage1":[ids], "stage2":[ids], ...}

split       = "validation"                          # "train" / "validation" / "test"
out_prefix  = "../../truthfulQA/truthfulQA_continual_setting/csqa_to_truthqa_top3_"             
topk        = 3
chunk       = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer(
    f"../../models/{baseline_model}_contrastive_model_{ablation_files[ablation]}",
    device=device,
)

with open(truth_file, encoding="utf-8") as f:
    truth_data = json.load(f)

truth_questions = [ex["question"] for ex in truth_data]
truth_ids       = [ex["id"]       for ex in truth_data]

truth_embs = model.encode(
    truth_questions,
    convert_to_tensor=True,
    batch_size=64,
    normalize_embeddings=True,
    device=device,
)


id2idx = {tid: i for i, tid in enumerate(truth_ids)}


with open(stage_file, encoding="utf-8") as f:
    stage_map = json.load(f)

stage1_ids = set(stage_map.get("stage1", []))
stage2_ids = set(stage_map.get("stage2", []))
stage3_ids = set(stage_map.get("stage3", []))

allowed_sets = [
    ("stage1",               stage1_ids),
    ("stage1_2",             stage1_ids | stage2_ids),
    ("stage1_2_3",           stage1_ids | stage2_ids | stage3_ids),
]


csqa         = load_dataset("tau/commonsense_qa", split=split)
cs_questions = [ex["question"] for ex in csqa]
cs_ids       = [ex["id"]       for ex in csqa]


cs_embs = []
for i in tqdm(range(0, len(cs_questions), chunk), desc="Embedding CSQA"):
    batch_qs = cs_questions[i : i + chunk]
    batch_embs = model.encode(
        batch_qs,
        convert_to_tensor=True,
        batch_size=64,
        normalize_embeddings=True,
        device=device,
    )
    cs_embs.append(batch_embs)
cs_embs = torch.cat(cs_embs, dim=0)               


for tag, allowed in allowed_sets:
    print(f"\n▶ Building mapping for {tag} ({len(allowed):,} allowed ids)")


    keep_idx   = [id2idx[tid] for tid in allowed if tid in id2idx]
    if len(keep_idx) < topk:
        raise ValueError(f"{tag}: allowed ids ({len(keep_idx)}) < topk ({topk})")

    sub_embs   = truth_embs[keep_idx]                 
    sub_ids    = [truth_ids[i] for i in keep_idx]

    mapping = {}
    
    sims     = cs_embs @ sub_embs.T                  
    values, idxs = torch.topk(sims, k=topk, dim=1)    

    for qid, row_idx, row_val in zip(cs_ids, idxs, values):
        mapping[qid] = {
            "truthfulQA_top3_ids"   : [sub_ids[int(j)] for j in row_idx],
            "truthfulQA_top3_cossim": [float(v)        for v in row_val],
        }


    assert len(mapping) == len(cs_ids), "Some CSQA questions were skipped!"


    out_file = f"{out_prefix}{tag}_{ablation_files[ablation]}.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"  ✔ saved → {out_file}  ({len(mapping):,} mappings)")

print("\nAll three stage-restricted mapping files created.")
