#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${CONFIG:-configs/graphbert_wikitext103.yaml}"
SOURCE_MODEL="${SOURCE_MODEL:-allenai/longformer-base-4096}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mldr-retrieval}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
FP16_FLAG="${FP16_FLAG:---fp16}"

SINGLE_OOD="${OUTPUT_ROOT}/single-vector-msmarco"
SINGLE_ID="${OUTPUT_ROOT}/single-vector-mldr"
COLBERT_OOD="${OUTPUT_ROOT}/colbert-msmarco"

# 1. Single Vector - Out Of Domain: MS MARCO hard-negative training.
python scripts/train_retrieval.py \
  --config "${CONFIG}" \
  --source-model "${SOURCE_MODEL}" \
  --output-dir "${SINGLE_OOD}" \
  --stage msmarco \
  --architecture single \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LR}" \
  --warmup-ratio 0.05 \
  --max-samples 1250000 \
  --gradient-checkpointing \
  ${FP16_FLAG}

python scripts/evaluate_mldr.py \
  --checkpoint "${SINGLE_OOD}" \
  --output-dir "${OUTPUT_ROOT}/single-vector-ood-eval" \
  --document-max-length 4096

# 2. Single Vector - In Domain: continue on the 10k MLDR English train queries.
python scripts/train_retrieval.py \
  --config "${CONFIG}" \
  --source-model "${SINGLE_OOD}" \
  --output-dir "${SINGLE_ID}" \
  --stage mldr \
  --architecture single \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LR}" \
  --warmup-ratio 0.05 \
  --max-samples 10000 \
  --document-max-length 4096 \
  --gradient-checkpointing \
  ${FP16_FLAG}

python scripts/evaluate_mldr.py \
  --checkpoint "${SINGLE_ID}" \
  --output-dir "${OUTPUT_ROOT}/single-vector-id-eval" \
  --document-max-length 4096

# 3. Multi Vector - Out Of Domain: MS MARCO ColBERT training, then MLDR PLAID retrieval.
python scripts/train_retrieval.py \
  --config "${CONFIG}" \
  --source-model "${SOURCE_MODEL}" \
  --output-dir "${COLBERT_OOD}" \
  --stage msmarco \
  --architecture colbert \
  --projection-dim 128 \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LR}" \
  --warmup-ratio 0.05 \
  --max-samples 810000 \
  --gradient-checkpointing \
  ${FP16_FLAG}

python scripts/evaluate_mldr_colbert.py \
  --checkpoint "${COLBERT_OOD}" \
  --output-dir "${OUTPUT_ROOT}/colbert-ood-eval" \
  --index-folder "${OUTPUT_ROOT}/colbert-index" \
  --document-max-length 4096