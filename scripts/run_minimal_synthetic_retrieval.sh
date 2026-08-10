#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${CONFIG:-configs/graphbert_wikitext103.yaml}"
SOURCE_MODEL="${SOURCE_MODEL:-allenai/longformer-base-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/synthetic-retrieval/appnp-100-steps}"
EVAL_DIR="${EVAL_DIR:-outputs/synthetic-retrieval/eval}"

python scripts/train_retrieval.py \
  --config "${CONFIG}" \
  --source-model "${SOURCE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stage synthetic \
  --architecture single \
  --pooling mean \
  --trainable-mode adapters \
  --synthetic-samples 512 \
  --synthetic-document-words 550 \
  --document-max-length 1024 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --epochs 10 \
  --max-steps 100 \
  --learning-rate 5e-4 \
  --warmup-ratio 0.1 \
  --gradient-checkpointing \
  --fp16

python scripts/evaluate_synthetic_retrieval.py \
  --checkpoint "${OUTPUT_DIR}" \
  --output-dir "${EVAL_DIR}" \
  --lengths 512 1024 2048 4096 \
  --num-queries 60 \
  --positions 0.1 0.5 0.9
