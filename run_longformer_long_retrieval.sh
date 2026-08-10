#!/usr/bin/env bash
#SBATCH --job-name=longformer-long-retrieval
#SBATCH --account=g.alex116u1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/venv}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_DIR}/outputs/sequential-search/baseline_longformer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/synthetic-retrieval-comparison}"
MAX_STEPS="${MAX_STEPS:-100}"
RUN_DIR="${OUTPUT_ROOT}/longformer"
CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint-${MAX_STEPS}"

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"
mkdir -p "${CHECKPOINT}" "${RUN_DIR}/results" "${RUN_DIR}/logs"

SOURCE_MODEL="${SOURCE_MODEL:-$(python scripts/latest_checkpoint.py \
  --root "${SOURCE_ROOT}" \
  --fallback allenai/longformer-base-4096)}"
printf '%s\n' "${SOURCE_MODEL}" > "${RUN_DIR}/source_checkpoint.txt"

python scripts/train_retrieval.py \
  --config configs/longformer_baseline.yaml \
  --source-model "${SOURCE_MODEL}" \
  --output-dir "${CHECKPOINT}" \
  --stage synthetic \
  --architecture single \
  --single-projection-dim 128 \
  --trainable-mode head \
  --synthetic-samples 512 \
  --synthetic-document-words 550 \
  --document-max-length 1024 \
  --batch-size 2 \
  --gradient-accumulation-steps 2 \
  --epochs 10 \
  --max-steps "${MAX_STEPS}" \
  --learning-rate 5e-4 \
  --warmup-ratio 0.1 \
  --device cuda \
  --fp16 \
  --gradient-checkpointing \
  2>&1 | tee "${RUN_DIR}/logs/train.log"

python scripts/evaluate_synthetic_retrieval.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_DIR}/results" \
  --lengths 512 1024 2048 4096 \
  --num-queries 60 \
  --positions 0.1 0.5 0.9 \
  --device cuda \
  2>&1 | tee "${RUN_DIR}/logs/evaluate.log"

python scripts/compare_synthetic_results.py --comparison-root "${OUTPUT_ROOT}" \
  | tee "${RUN_DIR}/logs/compare.log"
