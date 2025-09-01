import json, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ────────────────────────────────
# 0) 설정값 – 필요에 따라 수정
# ────────────────────────────────
# forget_file = "/home/work/data/seyun_workspace_home/cache_LTE/TOFU_ID/forget123.json"          
# dataset_files = [                           # 매핑을 만들 데이터셋들
#     {"path": "/home/work/data/seyun_workspace_home/cache_LTE/TOFU_ID/forget123.json", "question_key": "paraphrased_question"},
#     {"path": "/home/work/data/seyun_workspace_home/cache_LTE/TOFU_ID/retain_perturbed.json", "question_key": "question"},
#     {"path": "/home/work/data/seyun_workspace_home/cache_LTE/TOFU_ID/real_authors.json", "question_key": "question"},
#     {"path": "/home/work/data/seyun_workspace_home/cache_LTE/TOFU_ID/world_facts.json", "question_key": "question"},
# ]


stage = "1" # "1", "12", "123"

ablation = 0 # 0, 1, 2, 3, 4, 5

ablation_files = [
    "NQ_CURE_12K_a",
    "NQ_CURE_18K_a",
    "NQ_CURE_18K_a_no_b",
    "NQ_CURE_NO_HN_18K_a",
    "NQ_CURE_NO_HN_18K_a_no_b",
    "TQ_CURE_18K_a",
]

forget_file = f"../TOFU_NEW/stage{stage[-1]}/forget{stage}.json"         # NOTE!  DB 역할을 하는 파일,  아래 보시면 키가 question 입니다.

dataset_files = [                           # 매핑을 만들 데이터셋들
    {"path": f"../TOFU_NEW/stage{stage[-1]}/forget{stage}.json", "question_key": "paraphrased_question"},  # key 유의
    {"path": f"../TOFU_NEW/stage{stage[-1]}/forget{stage}_NU.json", "question_key": "question"},
    {"path": f"../TOFU_NEW/stage{stage[-1]}/retain_perturbed.json", "question_key": "question"},
    {"path": f"../TOFU_NEW/stage{stage[-1]}/real_authors.json", "question_key": "question"},
    {"path": f"../TOFU_NEW/stage{stage[-1]}/world_facts.json", "question_key": "question"},
]


out_file = f"../TOFU_NEW/stage{stage[-1]}/TOFU_to_forget{stage}_top3_{ablation_files[ablation]}.json"    # None 으로 두면 저장 생략
topk   = 3
chunk  = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer(
    f"../models/mpnet_contrastive_model_{ablation_files[ablation]}",
    device=device,
)

# ────────────────────────────────
# 1) forget data 로드 & 임베딩
# ────────────────────────────────
with open(forget_file, encoding="utf-8") as f:
    forget_data = json.load(f)

forget_questions = [ex["question"] for ex in forget_data]
forget_ids       = [ex["id"] for ex in forget_data]

forget_embs = model.encode(
    forget_questions,
    convert_to_tensor=True,
    device=device,
    batch_size=64,
    normalize_embeddings=True,
)

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
                "forget_data_top3_ids"   : [forget_ids[int(j)] for j in row_idx],
                "forget_data_top3_cossim": [float(v) for v in row_val],
            }

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


# missing_ids = []

# for cfg in dataset_files:
#     path = cfg["path"]
#     with open(path, encoding="utf-8") as f:
#         data = json.load(f)

#     for ex in data:
#         qid = ex["id"]
#         question = ex.get(cfg.get("question_key", "question"), ex.get("question"))
#         if qid not in mapping:
#             missing_ids.append({"id": qid, "question": question})

# # 누락된 질문 출력
# print(f"\n❌ 누락된 질문 수: {len(missing_ids)}")
# for m in missing_ids:
#     print(f"- ID: {m['id']}, Question: {m['question']}")





# from collections import Counter

# all_ids = []
# for cfg in dataset_files:
#     with open(cfg["path"], encoding="utf-8") as f:
#         data = json.load(f)
#     for ex in data:
#         all_ids.append(ex["id"])

# id_counts = Counter(all_ids)
# duplicates_across_files = [id for id, count in id_counts.items() if count > 1]

# print(f"⚠️ 여러 파일에 걸쳐 중복된 ID 수: {len(duplicates_across_files)}")
# if duplicates_across_files:
#     print("예시 중복 ID들:", duplicates_across_files[:5])

# all_ids = set()
# for cfg in dataset_files:
#     with open(cfg["path"], encoding="utf-8") as f:
#         data = json.load(f)
#     for ex in data:
#         all_ids.add(ex["id"])

# mapped_ids = set(mapping.keys())
# missing_ids = all_ids - mapped_ids
# print(f"누락된 ID들: {missing_ids}")


# seen_ids = set()
# for ex in data:
#     qid = ex["id"]
#     if qid in seen_ids:
#         print(f"⚠️ 중복된 ID 발견: {qid} in {path}")
#     seen_ids.add(qid)
