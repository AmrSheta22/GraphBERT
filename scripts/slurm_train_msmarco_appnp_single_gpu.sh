#!/usr/bin/env bash
#SBATCH --job-name=appnp-msmarco
#SBATCH --account=g.alex116
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err

set -Eeuo pipefail

# Edit the account above before submitting:
#   #SBATCH --account=g.<your_ba_hpc_project>
#
# This starts from the original Longformer weights and adds APPNP adapters from
# the graph section of BASE_CONFIG. No MLM checkpoint is required.
#
# Submit from the repository root:
#   sbatch scripts/slurm_train_msmarco_appnp_single_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${CONFIG:-configs/graphbert_wikitext103.yaml}"
SOURCE_MODEL="${SOURCE_MODEL:-allenai/longformer-base-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mldr/appnp-from-longformer-msmarco-single-gpu}"
ARCHITECTURE="${ARCHITECTURE:-single}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1250000}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-64}"
DOCUMENT_MAX_LENGTH="${DOCUMENT_MAX_LENGTH:-512}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
FP16_FLAG="${FP16_FLAG:---fp16}"
GRADIENT_CHECKPOINTING_FLAG="${GRADIENT_CHECKPOINTING_FLAG:---gradient-checkpointing}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/venv}"

cd "${REPO_DIR}"

# Uncomment or edit these lines to match the modules available on BA-HPC.
# module clear
# module load CUDA
# module load python

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
elif [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  source "${REPO_DIR}/.venv/bin/activate"
elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
  source "${VENV_DIR}/Scripts/activate"
else
  echo "Could not find a virtual environment activation script." >&2
  echo "Looked for: ${VENV_DIR}/bin/activate, ${REPO_DIR}/.venv/bin/activate, ${VENV_DIR}/Scripts/activate" >&2
  exit 1
fi

echo "Job started at: $(date)"
echo "Run: Longformer + APPNP adapters -> MS MARCO"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Config: ${CONFIG}"
echo "Source model: ${SOURCE_MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python executable: $(command -v python)"
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
