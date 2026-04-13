import json, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import sys


model_size = "7B"  # "1B", "7B"
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


if model_size == "1B":
    data_folder = "Meta-Llama-3.2-1B-Instruct_dataset"
elif model_size == "7B":
    data_folder = "Meta-Llama-2-7B-chat_dataset"

db_times = []
db_sizes = []
search_times = []
search_sizes = []

for i in range(9, 10):

    forget_file = f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_forget.json"
    dataset_files = [                          
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_forget_paraphrased.json", "question_key": "paraphrased_instruction"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_retain_used.json", "question_key": "question"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_retain_not_used.json", "question_key": "question"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/non_target.json", "question_key": "question"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_near_utility.json", "question_key": "contrastive_instruction"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/winogrande_xs_validation.json", "question_key": "sentence"},
    ]


    out_file = f"../RETURN_NEW_DATASET/{data_folder}/RETURN_stage_{i}_top3_{ablation_files[ablation]}.json"   
    topk   = 3
  
    chunk = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(
        f"../models/mpnet_contrastive_model_{ablation_files[ablation]}",
        device=device,
    )


    with open(forget_file, encoding="utf-8") as f:
        forget_data = json.load(f)

    torch.cuda.synchronize()
    start_time = time.time()
    
    forget_questions = [ex["question"] for ex in forget_data]
    forget_ids       = [ex["id"] for ex in forget_data]



    forget_embs = model.encode(
        forget_questions,
        convert_to_tensor=True,
        device=device,
        batch_size=30,
        normalize_embeddings=True,
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    # take the last one only
    if i == 9:
        db_times.append(end_time - start_time)
        db_sizes.append(len(forget_questions))

 
    mapping = {}
    for cfg in dataset_files:
        path   = cfg["path"]
        q_key  = cfg.get("question_key", "question")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        qs, qids = [], []
        for ex in data:
            question = ex.get(q_key) or ex.get("question")
            if question is None:
                raise KeyError(f"No '{q_key}' or 'question' field found in the example at {path}.")
            qs.append(question)
            qids.append(ex["id"])
        
        torch.cuda.synchronize()
        start_time = time.time()

        # print("len qs: ", len(qs))
        for j in tqdm(range(0, len(qs), chunk), desc=f"Embedding {Path(path).name}"):
            batch_qs  = qs[j : j + chunk]
            batch_ids = qids[j : j + chunk]
            # print("chunk: ", chunk)
            # print("batch qs:", batch_qs)

            batch_embs = model.encode(
                batch_qs,
                convert_to_tensor=True,
                device=device,
                batch_size=64,
                normalize_embeddings=True,
            )

            sims          = batch_embs @ forget_embs.T
            values, idxs  = torch.topk(sims, k=topk, dim=1)

            for qid, row_idx, row_val in zip(batch_ids, idxs, values):
                mapping[qid] = {
                    "forget_data_top3_ids"   : [forget_ids[int(k)] for k in row_idx],
                    "forget_data_top3_cossim": [float(v) for v in row_val],
                }
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # search_times.append((i, end_time - start_time))
        # search_sizes.append((i, len(qs)))
        if i == 9:
            search_times.append(end_time - start_time)
            search_sizes.append(len(qs))

print("db_times:", db_times)
print("db_sizes:", db_sizes)
print("search_times:", search_times)
print("search_sizes:", search_sizes)
average_search_times = [t / s for t, s in zip(search_times, search_sizes)]
print("Average Unlearning Time: ", db_times[0] / 10)
print("Average Search Time per Query: ", sum(average_search_times) / len(average_search_times))
