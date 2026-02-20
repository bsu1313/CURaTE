#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# FAISS + SentenceTransformer benchmark runner
# Usage:
#   bash faiss_run.sh 200000
#   bash faiss_run.sh 500000
#   bash faiss_run.sh 1000000
# Optional:
#   DEVICE=cpu bash faiss_run.sh 200000
# ==========================================

# -------------------------------
# CONFIG
# -------------------------------
ENV_NAME="faiss_bench"
SCRIPT="faiss_with_unrelated_corpus.py"

MODEL_NAME="models/mpnet_contrastive_model_NQ_CURE_18K_a"
FORGET_JSON="RETURN_NEW_DATASET/Meta-Llama-2-7B-chat_dataset/stage_9_forget_paraphrased.json"
FORGET_KEY="paraphrased_instruction"

UNREL_DATASET="wikimedia/wikipedia"
UNREL_CONFIG="20231101.en"
UNREL_SPLIT="train"
UNREL_TEXT_FIELD="text"

# Scale (corpus size)
UNREL_DOCS=${1:-1000000}

# Text truncation for unrelated corpus docs
MAX_CHARS=${MAX_CHARS:-400}

# Query settings
QUERY_N=${QUERY_N:-2000}
TOPK=${TOPK:-10}

# Encoding settings
BATCH_SIZE=${BATCH_SIZE:-512}

# Device: cuda (default) or cpu
DEVICE=${DEVICE:-cuda}

# -------------------------------
# Derived params: NLIST/TRAIN_SIZE
# -------------------------------
# Good rule of thumb: nlist ~ sqrt(N), train_size ~ min(200k, N)
if [ "${UNREL_DOCS}" -le 200000 ]; then
  NLIST=${NLIST:-1024}
  TRAIN_SIZE=${TRAIN_SIZE:-100000}
elif [ "${UNREL_DOCS}" -le 500000 ]; then
  NLIST=${NLIST:-2048}
  TRAIN_SIZE=${TRAIN_SIZE:-200000}
else
  NLIST=${NLIST:-4096}
  TRAIN_SIZE=${TRAIN_SIZE:-200000}
fi

# IVF search setting
NPROBE=${NPROBE:-16}

# fp16 only makes sense on cuda
FP16_FLAG=""
if [ "${DEVICE}" = "cuda" ]; then
  FP16_FLAG="--fp16"
fi

# -------------------------------
# RUNTIME
# -------------------------------
echo "Activating conda environment: ${ENV_NAME}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

# Avoid import/path contamination
unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "Starting FAISS benchmark..."
echo "Corpus size (unrelated docs): ${UNREL_DOCS}"
echo "Device: ${DEVICE}"
echo "IVF params: nlist=${NLIST}, nprobe=${NPROBE}, train_size=${TRAIN_SIZE}"
echo "Encode bs: ${BATCH_SIZE}, query_n=${QUERY_N}, topk=${TOPK}, max_chars=${MAX_CHARS}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="faiss_benchmark_N${UNREL_DOCS}_nlist${NLIST}_nprobe${NPROBE}_${DEVICE}_${TIMESTAMP}.log"

python "${SCRIPT}" \
  --model_name "${MODEL_NAME}" \
  --forget_json "${FORGET_JSON}" \
  --forget_key "${FORGET_KEY}" \
  --unrel_dataset "${UNREL_DATASET}" \
  --unrel_config "${UNREL_CONFIG}" \
  --unrel_split "${UNREL_SPLIT}" \
  --unrel_text_field "${UNREL_TEXT_FIELD}" \
  --unrel_docs "${UNREL_DOCS}" \
  --max_chars "${MAX_CHARS}" \
  --encode_bs "${BATCH_SIZE}" \
  ${FP16_FLAG} \
  --device "${DEVICE}" \
  --query_n "${QUERY_N}" \
  --k "${TOPK}" \
  --nlist "${NLIST}" \
  --nprobe "${NPROBE}" \
  --train_size "${TRAIN_SIZE}" \
  2>&1 | tee "${LOGFILE}"

echo "Finished."
echo "Log saved to ${LOGFILE}"