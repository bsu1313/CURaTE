import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# 1. 준비 – 모델과 데이터, stage ID 세트 로드
# ────────────────────────────────────────────────────────────────

baseline_model = "mpnet" # mpnet, minilm, distilroberta
ablation = 1 # 0, 1, 2, 3, 4, 5, 6

ablation_files = [
    "NQ_CURE_12K_a",
    "NQ_CURE_18K_a",
    "NQ_CURE_18K_a_no_b",
    "NQ_CURE_NO_HN_18K_a",
    "NQ_CURE_NO_HN_18K_a_no_b",
    "TQ_CURE_18K_a",
    "no_finetuning"
]

model = SentenceTransformer(f"../../models/{baseline_model}_contrastive_model_{ablation_files[ablation]}")

with open("../../truthfulQA/truthfulQA_all_augmented_ID.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ❶ stage 정보가 들어있는 JSON 파일
#    예: {"stage1":[1,2,3], "stage2":[...], "stage3":[...]}
with open("../../truthfulQA/truthfulQA_continual_setting/TruthfulQA_split_ids.json", "r", encoding="utf-8") as f:
    stages = json.load(f)

# ↘️ stage 집합 준비
stage_sets = {
    "stage1":                set(stages["stage1"]),
    "stage1_2":              set(stages["stage1"])      | set(stages["stage2"]),
    "stage1_2_3":            set(stages["stage1"])      | set(stages["stage2"]) | set(stages["stage3"]),
}

# 데이터 전체 임베딩은 한 번만!
all_questions = [item["question"] for item in data]
question_embeddings = model.encode(all_questions, convert_to_tensor=True)

# id → (원본 index, embedding index) 매핑
id2idx = {item["id"]: i for i, item in enumerate(data)}

# ────────────────────────────────────────────────────────────────
# 2. 공통 처리 함수를 정의
# ────────────────────────────────────────────────────────────────
def run_similarity(stage_tag: str, allowed_ids: set):
    """
    한 개 stage 범위에 대해 모든 통계를 계산하고 두 개의 파일을 저장한다.
    """
    allowed_indices = np.array([id2idx[i] for i in allowed_ids if i in id2idx])

    # stage 범위 안에 있는 질문만 대상으로 accuracy를 집계할 것
    total = len(allowed_indices)

    results               = []
    paraphrased_scores    = []
    contrastive_scores    = []
    topk_stats = {
        "paraphrased": {1: 0, 2: 0, 3: 0, "missed_ids": {1: [], 2: [], 3: []}},
        "contrastive": {1: 0, 2: 0, 3: 0, "missed_ids": {1: [], 2: [], 3: []}},
    }
    top3_id_mapping = {}

    # stage 범위 질문 한 바퀴
    for idx in tqdm(allowed_indices, desc=f"▶ Stage {stage_tag}"):
        item = data[idx]

        # ── paraphrased
        paraphrased_emb   = model.encode(item["paraphrased_question"], convert_to_tensor=True)
        sim_vec           = util.cos_sim(paraphrased_emb, question_embeddings)[0].cpu().numpy()
        sims_sub          = sim_vec[allowed_indices]                      # stage 제한
        sub_idx_sorted    = allowed_indices[np.argsort(-sims_sub)[:3]]    # top3 원본 index
        paraphrased_scores.append(float(sim_vec[idx]))

        for k in (1, 2, 3):
            if idx in sub_idx_sorted[:k]:
                topk_stats["paraphrased"][k] += 1
            else:
                topk_stats["paraphrased"]["missed_ids"][k].append(item["id"])

        # ── contrastive
        contrastive_emb   = model.encode(item["contrastive_question"], convert_to_tensor=True)
        sim_vec2          = util.cos_sim(contrastive_emb, question_embeddings)[0].cpu().numpy()
        sims_sub2         = sim_vec2[allowed_indices]
        sub_idx_sorted2   = allowed_indices[np.argsort(-sims_sub2)[:3]]
        contrastive_scores.append(float(sim_vec2[idx]))

        for k in (1, 2, 3):
            if idx in sub_idx_sorted2[:k]:
                topk_stats["contrastive"][k] += 1
            else:
                topk_stats["contrastive"]["missed_ids"][k].append(item["id"])

        # ── 기록
        top3_id_mapping[item["id"]] = {
            "paraphrased_top3_ids":   [data[i]["id"] for i in sub_idx_sorted],
            "paraphrased_top3_cos":   [float(sim_vec[i]) for i in sub_idx_sorted],
            "contrastive_top3_ids":   [data[i]["id"] for i in sub_idx_sorted2],
            "contrastive_top3_cos":   [float(sim_vec2[i]) for i in sub_idx_sorted2],
        }

        results.append({
            "id": item["id"],
            "paraphrased_question_similarity_top3": [
                {
                    "question_index": int(i),
                    "question_text" : data[i]["question"],
                    "score": float(sim_vec[i]),
                } for i in sub_idx_sorted
            ],
            "contrastive_question_similarity_top3": [
                {
                    "question_index": int(i),
                    "question_text" : data[i]["question"],
                    "score": float(sim_vec2[i]),
                } for i in sub_idx_sorted2
            ],
            "self_similarity": {
                "paraphrased_to_question": float(sim_vec[idx]),
                "contrastive_to_question": float(sim_vec2[idx]),
            },
        })

    # ── stage 통계
    statistics = {
        "paraphrased": {
            "top1_accuracy"    : topk_stats["paraphrased"][1] / total,
            "top2_accuracy"    : topk_stats["paraphrased"][2] / total,
            "top3_accuracy"    : topk_stats["paraphrased"][3] / total,
            "average_similarity": float(np.mean(paraphrased_scores)),
            "missed_ids"       : topk_stats["paraphrased"]["missed_ids"],
        },
        "contrastive": {
            "top1_accuracy"    : topk_stats["contrastive"][1] / total,
            "top2_accuracy"    : topk_stats["contrastive"][2] / total,
            "top3_accuracy"    : topk_stats["contrastive"][3] / total,
            "average_similarity": float(np.mean(contrastive_scores)),
            "missed_ids"       : topk_stats["contrastive"]["missed_ids"],
        },
    }

    # ──────────────────────────────────────────────────────────
    # 3. 파일 저장
    # ──────────────────────────────────────────────────────────
    # out_dir = Path(".")
    out_dir = Path("../../truthfulQA/truthfulQA_continual_setting/")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"similarity_results_{stage_tag}_{ablation_files[ablation]}.json", "w", encoding="utf-8") as f:
        json.dump({"results":results, "statistics":statistics}, f, indent=2, ensure_ascii=False)

    with open(out_dir / f"top3_id_mappings_{stage_tag}_{ablation_files[ablation]}.json", "w", encoding="utf-8") as f:
        json.dump(top3_id_mapping, f, indent=2, ensure_ascii=False)

    print(f"✅ Stage {stage_tag}: 저장 완료!")

# ────────────────────────────────────────────────────────────────
# 4. 모든 stage 범위에 대해 실행
# ────────────────────────────────────────────────────────────────
for tag, ids in stage_sets.items():
    run_similarity(tag, ids)
