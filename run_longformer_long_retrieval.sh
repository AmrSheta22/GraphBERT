#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

BASE_CONFIG="${BASE_CONFIG:-configs/graphbert_wikitext103.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/sequential-search/baseline_longformer}"
SOURCE_MODEL_FALLBACK="${SOURCE_MODEL_FALLBACK:-allenai/longformer-base-4096}"
COMPARISON_ROOT="${COMPARISON_ROOT:-outputs/synthetic-retrieval-comparison}"
RUN_ROOT="${COMPARISON_ROOT}/longformer"
MAX_STEPS="${MAX_STEPS:-100}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints/checkpoint-${MAX_STEPS}"
RESULTS_DIR="${RUN_ROOT}/results"
CONFIG="${RUN_ROOT}/longformer_config.yaml"
DEVICE="${DEVICE:-cuda}"

resolve_source_model() {
  if [[ -n "${SOURCE_MODEL:-}" ]]; then
    printf '%s\n' "${SOURCE_MODEL}"
    return
  fi
  python - "${SOURCE_ROOT}" "${SOURCE_MODEL_FALLBACK}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
fallback = sys.argv[2]
weight_names = ("model.safetensors", "pytorch_model.bin")
candidates = []
if root.exists():
    for path in root.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir() and any((path / name).exists() for name in weight_names):
            candidates.append((int(match.group(1)), path))
if candidates:
    print(max(candidates)[1])
elif any((root / name).exists() for name in weight_names):
    print(root)
else:
    print(fallback)
PY
}

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${RUN_ROOT}/logs"
python - "${BASE_CONFIG}" "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

source, target = map(Path, sys.argv[1:])
with source.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config["graph"]["num_replaced_layers"] = 0
config["graph"]["layer_indices"] = []
with target.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

SOURCE_MODEL_RESOLVED="$(resolve_source_model)"
printf '%s\n' "${SOURCE_MODEL_RESOLVED}" > "${RUN_ROOT}/source_checkpoint.txt"

FP16_ARGS=()
CHECKPOINTING_ARGS=()
if [[ "${DEVICE}" == "cuda" ]]; then
  FP16_ARGS=(--fp16)
  CHECKPOINTING_ARGS=(--gradient-checkpointing)
fi

echo "Longformer source checkpoint: ${SOURCE_MODEL_RESOLVED}"
echo "Longformer output checkpoint: ${CHECKPOINT_DIR}"

python scripts/train_retrieval.py \
  --config "${CONFIG}" \
  --source-model "${SOURCE_MODEL_RESOLVED}" \
  --output-dir "${CHECKPOINT_DIR}" \
  --stage synthetic \
  --architecture single \
  --pooling mean \
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
  --device "${DEVICE}" \
  "${FP16_ARGS[@]}" \
  "${CHECKPOINTING_ARGS[@]}" \
  2>&1 | tee "${RUN_ROOT}/logs/train.log"

python scripts/evaluate_synthetic_retrieval.py \
  --checkpoint "${CHECKPOINT_DIR}" \
  --output-dir "${RESULTS_DIR}" \
  --lengths 512 1024 2048 4096 \
  --num-queries 60 \
  --positions 0.1 0.5 0.9 \
  --device "${DEVICE}" \
  2>&1 | tee "${RUN_ROOT}/logs/evaluate.log"

python scripts/compare_synthetic_results.py --comparison-root "${COMPARISON_ROOT}" \
  | tee "${RUN_ROOT}/logs/compare.log"
