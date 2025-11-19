import json, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import numpy as np
import time


model_size = "1B"  # "1B", "7B"
baseline_model = "mpnet" # mpnet, minilm, distilroberta
ablation = 1 # 0, 1, 2, 3, 4, 5, 6

ablation_files = [
    "NQ_CURE_12K_a",
    "NQ_CURE_18K_a",
    "NQ_CURE_18K_a_no_b",
    "NQ_CURE_NO_HN_18K_a",
    "NQ_CURE_NO_HN_18K_a_no_b",
    "TQ_CURE_18K_a",
    "no_finetuning",
    "guard"
]


if model_size == "1B":
    data_folder = "Meta-Llama-3.2-1B-Instruct_dataset"
elif model_size == "7B":
    data_folder = "Meta-Llama-2-7B-chat_dataset"

db_times = []
db_sizes = []
search_times = []
search_sizes = []

for i in range(10):

    forget_file = f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_forget.json"
    dataset_files = [                           # 매핑을 만들 데이터셋들
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_forget_paraphrased.json", "question_key": "paraphrased_instruction"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_retain_used.json", "question_key": "question"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_retain_not_used.json", "question_key": "question"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/non_target.json", "question_key": "question"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/stage_{i}_near_utility.json", "question_key": "contrastive_instruction"},
        {"path": f"../RETURN_NEW_DATASET/{data_folder}/winogrande_xs_validation.json", "question_key": "sentence"},
    ]


    out_file = f"../RETURN_NEW_DATASET/{data_folder}/RETURN_stage_{i}_top3_{ablation_files[ablation]}.json"    # None 으로 두면 저장 생략
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

    # ────────────────────────────────
    # 1) forget data 로드 & 임베딩
    # ────────────────────────────────
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
        batch_size=64,
        normalize_embeddings=True,
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    db_times.append(end_time - start_time)
    db_sizes.append(len(forget_questions))

    # ────────────────────────────────
    # 2) 각 데이터셋 처리
    # ────────────────────────────────
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
                raise KeyError(f"{path} 예시에 '{q_key}' 또는 'question' 필드가 없습니다.")
            qs.append(question)
            qids.append(ex["id"])

        for i in tqdm(range(0, len(qs), chunk), desc=f"Embedding {Path(path).name}"):
            batch_qs  = qs[i : i + chunk]
            batch_ids = qids[i : i + chunk]
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

            for qid, q_text, row_idx, row_val in zip(batch_ids, batch_qs, idxs, values):

                if ablation == 7:
                    # take the cosine top-5 candidates
                    cand_idx = row_idx[:5].tolist()

                    # build (query, candidate) pairs for the CrossEncoder
                    pairs = [(q_text, forget_questions[j]) for j in cand_idx]

                    with torch.inference_mode():
                        ce_scores = reranker.predict(pairs)  # higher is more similar

                    # pick new top-3 under CE scores
                    order = np.argsort(ce_scores)[::-1][:3]
                    reranked_ids    = [forget_ids[cand_idx[k]] for k in order]
                    reranked_scores = [float(ce_scores[k]) for k in order]

                    # (optional) also keep original cosine scores for those chosen
                    # map CE-selected indices back to their cosine scores
                    cos_for_reranked = []
                    for k in order:
                        j_global = cand_idx[k]
                        # find where j_global sits in row_idx to fetch its cosine
                        pos = (row_idx == j_global).nonzero(as_tuple=True)[0].item()
                        cos_for_reranked.append(float(row_val[pos]))

                    mapping[qid] = {
                        "forget_data_top3_ids": reranked_ids,
                        "forget_data_top3_crossenc": reranked_scores,
                        "forget_data_top3_cossim": cos_for_reranked,
                    }

                else:
                    # original cosine top-k path (return top-3 to keep the same shape)
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

    # ────────────────────────────────
    # 3) 저장 및 검증
    # ────────────────────────────────
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    total_questions = sum(len(json.load(open(cfg["path"], encoding="utf-8")))
                        for cfg in dataset_files)
    if len(mapping) == total_questions:
        print("✅ 모든 질문이 매핑되었습니다.")
    else:
        print(f"⚠️ {total_questions - len(mapping)} 개의 질문이 누락되었습니다.")

    print(f"총 매핑 수: {len(mapping):,}")
    if out_file:
        print(f"Saved to → {out_file}")






from collections import Counter, defaultdict

id_counter  = Counter()
id_sources  = defaultdict(list)   # {id: [파일경로1, 파일경로2, ...]}

for cfg in dataset_files:
    path = cfg["path"]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for ex in data:
        qid = ex["id"]
        id_counter[qid] += 1
        id_sources[qid].append(path)

# ⚠️ 실제로 중복된 ID 목록 뽑기
dup_ids = [qid for qid, cnt in id_counter.items() if cnt > 1]

print(f"⚠️ 중복된 ID 개수: {len(dup_ids)}\n")
for qid in dup_ids:
    print(f"- {qid}  ←  {', '.join(id_sources[qid])}")
