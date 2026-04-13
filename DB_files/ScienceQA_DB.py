import json, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import numpy as np
import time


# "biology" -> "physics" -> "chemistry" -> "economics" 

stage = 4 # 1, 2, 3, 4
baseline_model = "mpnet" # mpnet, minilm, distilroberta
ablation = 1 # 0, 1, 2, 3, 4, 5, 6

ablation_files = [
    "NQ_CURaTE_12K_a",
    "NQ_CURaTE_18K_a",
    "NQ_CURaTE_18K_a_no_b",
    "NQ_CURaTE_NO_HN_18K_a",
    "NQ_CURaTE_NO_HN_18K_a_no_b",
    "TQ_CURaTE_18K_a",
    "no_finetuning",
    "guard"
]

if stage == 1:
    forget_file = "../ScienceQA/test_forget_PR/PR_scienceqa_biology_train_SD.json"         
    dataset_files = [                           
        {"path": "../ScienceQA/test_forget_PR/PR_scienceqa_biology_train_SD.json", "question_key": "paraphrased_instruction"},  
        {"path": "../ScienceQA/retain/processed_scienceqa_not_biology_test_RD.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_NU/NU_scienceqa_biology_train_SD.json", "question_key": "contrastive_instruction"}, 
        {"path": "../ScienceQA/test_utility/processed_commonqa_test.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_utility/processed_openbookqa_test.json", "question_key": "instruction"},
    ]
    out_file = f"../ScienceQA/ScienceQA_to_stage1_top3_{ablation_files[ablation]}.json"    
elif stage == 2:
    forget_file = "../ScienceQA/test_forget_PR/PR_scienceqa_biology_physics_train_SD.json"         
    dataset_files = [                           
        {"path": "../ScienceQA/test_forget_PR/PR_scienceqa_biology_physics_train_SD.json", "question_key": "paraphrased_instruction"},  
        {"path": "../ScienceQA/retain/processed_scienceqa_not_biology_physics_test_RD.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_NU/NU_scienceqa_biology_physics_train_SD.json", "question_key": "contrastive_instruction"}, 
        {"path": "../ScienceQA/test_utility/processed_commonqa_test.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_utility/processed_openbookqa_test.json", "question_key": "instruction"},
    ]
    out_file = f"../ScienceQA/ScienceQA_to_stage2_top3_{ablation_files[ablation]}.json"    
elif stage == 3:
    forget_file = "../ScienceQA/test_forget_PR/PR_scienceqa_biology_physics_chemistry_train_SD.json"         
    dataset_files = [                           
        {"path": "../ScienceQA/test_forget_PR/PR_scienceqa_biology_physics_chemistry_train_SD.json", "question_key": "paraphrased_instruction"},  
        {"path": "../ScienceQA/retain/processed_scienceqa_not_biology_physics_chemistry_test_RD.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_NU/NU_scienceqa_biology_physics_chemistry_train_SD.json", "question_key": "contrastive_instruction"}, 
        {"path": "../ScienceQA/test_utility/processed_commonqa_test.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_utility/processed_openbookqa_test.json", "question_key": "instruction"},
    ]
    out_file = f"../ScienceQA/ScienceQA_to_stage3_top3_{ablation_files[ablation]}.json"    
elif stage == 4:
    forget_file = "../ScienceQA/test_forget_PR/PR_scienceqa_biology_physics_chemistry_economics_train_SD.json"         
    dataset_files = [                           
        {"path": "../ScienceQA/test_forget_PR/PR_scienceqa_biology_physics_chemistry_economics_train_SD.json", "question_key": "paraphrased_instruction"},  
        {"path": "../ScienceQA/retain/processed_scienceqa_not_biology_physics_chemistry_economics_test_RD.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_NU/NU_scienceqa_biology_physics_chemistry_economics_train_SD.json", "question_key": "contrastive_instruction"}, 
        {"path": "../ScienceQA/test_utility/processed_commonqa_test.json", "question_key": "instruction"},
        {"path": "../ScienceQA/test_utility/processed_openbookqa_test.json", "question_key": "instruction"},
    ]
    out_file = f"../ScienceQA/ScienceQA_to_stage4_top3_{ablation_files[ablation]}.json"    


if ablation == 7:
    topk = 5
else:
    topk   = 3
chunk  = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
if ablation == 7:
    model  = SentenceTransformer(
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        device=device,
    )
    reranker = CrossEncoder("cross-encoder/stsb-roberta-base")
else:
    model  = SentenceTransformer(
        f"../models/{baseline_model}_contrastive_model_{ablation_files[ablation]}",
        device=device,
    )


with open(forget_file, encoding="utf-8") as f:
    forget_data = json.load(f)

forget_questions = [ex["instruction"] for ex in forget_data]
forget_ids       = [ex["id"] for ex in forget_data]

forget_embs = model.encode(
    forget_questions,
    convert_to_tensor=True,
    device=device,
    batch_size=64,
    normalize_embeddings=True,
)


mapping = {}
for cfg in dataset_files:
    path   = cfg["path"]
    q_key  = cfg.get("question_key")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    qs, qids = [], []
    for ex in data:
        question = ex.get(q_key) # or ex.get("question")
        if question is None:
            raise KeyError(f"No '{q_key}' or 'question' field found in the example at {path}.")
        qs.append(question)
        qids.append(ex["id"])

    for i in tqdm(range(0, len(qs), chunk), desc=f"Embedding {Path(path).name}"):
        batch_qs  = qs[i : i + chunk]
        batch_ids = qids[i : i + chunk]

        batch_embs = model.encode(
            batch_qs,
            convert_to_tensor=True,
            device=device,
            batch_size=64,
            normalize_embeddings=True,
        )

        sims          = batch_embs @ forget_embs.T
        values, idxs  = torch.topk(sims, k=topk, dim=1)

        for qid, q_text, row_idx, row_val in zip(batch_ids, batch_qs, idxs, values):

            if ablation == 7:
                # take the cosine top-5 candidates
                cand_idx = row_idx[:5].tolist()

               
                pairs = [(q_text, forget_questions[j]) for j in cand_idx]

                with torch.inference_mode():
                    ce_scores = reranker.predict(pairs)  # higher is more similar

           
                order = np.argsort(ce_scores)[::-1][:3]
                reranked_ids    = [forget_ids[cand_idx[k]] for k in order]
                reranked_scores = [float(ce_scores[k]) for k in order]


                cos_for_reranked = []
                for k in order:
                    j_global = cand_idx[k]
                 
                    pos = (row_idx == j_global).nonzero(as_tuple=True)[0].item()
                    cos_for_reranked.append(float(row_val[pos]))

                mapping[qid] = {
                    "forget_data_top3_ids": reranked_ids,
                    "forget_data_top3_crossenc": reranked_scores,
                    "forget_data_top3_cossim": cos_for_reranked,
                }

            else:

                top3_idx = row_idx[:3].tolist()
                top3_val = row_val[:3].tolist()
                mapping[qid] = {
                    "forget_data_top3_ids"   : [forget_ids[int(j)] for j in top3_idx],
                    "forget_data_top3_cossim": [float(v) for v in top3_val],
                }
        
        # for qid, row_idx, row_val in zip(batch_ids, idxs, values):
        #     mapping[qid] = {
        #         "forget_data_top3_ids"   : [forget_ids[int(j)] for j in row_idx],
        #         "forget_data_top3_cossim": [float(v) for v in row_val],
        #     }


if out_file:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

total_questions = sum(len(json.load(open(cfg["path"], encoding="utf-8")))
                      for cfg in dataset_files)
if len(mapping) == total_questions:
    print("Mapping complete!")
else:
    print(f"⚠️ {total_questions - len(mapping)} questions are missing.")

print(f"Total mappings: {len(mapping):,}")
if out_file:
    print(f"Saved to → {out_file}")








# from collections import Counter

# all_ids = []
# for cfg in dataset_files:
#     with open(cfg["path"], encoding="utf-8") as f:
#         data = json.load(f)
#     for ex in data:
#         all_ids.append(ex["id"])

# id_counts = Counter(all_ids)
# duplicates_across_files = [id for id, count in id_counts.items() if count > 1]

# print(f"⚠️ Number of duplicate IDs across files: {len(duplicates_across_files)}")
# if duplicates_across_files:
#     print("Example duplicate IDs:", duplicates_across_files[:5])

# all_ids = set()
# for cfg in dataset_files:
#     with open(cfg["path"], encoding="utf-8") as f:
#         data = json.load(f)
#     for ex in data:
#         all_ids.add(ex["id"])

# mapped_ids = set(mapping.keys())
# missing_ids = all_ids - mapped_ids
# print(f"Missing IDs: {missing_ids}")


# seen_ids = set()
# for ex in data:
#     qid = ex["id"]
#     if qid in seen_ids:
#         print(f"⚠️ Duplicate ID found: {qid} in {path}")
#     seen_ids.add(qid)
