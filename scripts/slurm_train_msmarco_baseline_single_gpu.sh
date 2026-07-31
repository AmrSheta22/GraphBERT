#!/usr/bin/env bash
#SBATCH --job-name=longformer-msmarco
#SBATCH --account=g.alex116u1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -Eeuo pipefail

# Edit the account above before submitting:
#   #SBATCH --account=g.<your_ba_hpc_project>
#
# Submit from the repository root:
#   sbatch scripts/slurm_train_msmarco_baseline_single_gpu.sh

REPO_DIR="${REPO_DIR:-$PWD}"
BASE_CONFIG="${BASE_CONFIG:-configs/graphbert_wikitext103.yaml}"
SOURCE_MODEL="${SOURCE_MODEL:-allenai/longformer-base-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mldr/baseline-msmarco-single-gpu}"
ARCHITECTURE="${ARCHITECTURE:-single}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1250000}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-64}"
DOCUMENT_MAX_LENGTH="${DOCUMENT_MAX_LENGTH:-512}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
FP16_FLAG="${FP16_FLAG:---fp16}"
GRADIENT_CHECKPOINTING_FLAG="${GRADIENT_CHECKPOINTING_FLAG:---gradient-checkpointing}"

cd "${REPO_DIR}"
mkdir -p outputs/slurm "${OUTPUT_DIR}"

# Uncomment or edit these lines to match the modules available on BA-HPC.
# module clear
# module load CUDA
# module load python

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
fi

CONFIG="${OUTPUT_DIR}/resolved_retrieval_train_config.yaml"
python - "${BASE_CONFIG}" "${CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, target_config = sys.argv[1:]
with Path(base_config).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["graph"]["num_replaced_layers"] = 0
config["graph"]["layer_indices"] = []

with Path(target_config).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

echo "Job started at: $(date)"
echo "Run: baseline Longformer -> MS MARCO"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: ${CONFIG}"
echo "Source model: ${SOURCE_MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
PY

python scripts/train_retrieval.py \
  --config "${CONFIG}" \
  --source-model "${SOURCE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --stage msmarco \
  --architecture "${ARCHITECTURE}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --max-samples "${MAX_SAMPLES}" \
  --query-max-length "${QUERY_MAX_LENGTH}" \
  --document-max-length "${DOCUMENT_MAX_LENGTH}" \
  --logging-steps "${LOGGING_STEPS}" \
  --device cuda \
  ${FP16_FLAG} \
  ${GRADIENT_CHECKPOINTING_FLAG}

echo "Job finished at: $(date)"
