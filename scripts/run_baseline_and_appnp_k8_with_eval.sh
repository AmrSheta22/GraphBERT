#!/usr/bin/env bash
set -Eeuo pipefail

BASE_CONFIG="${BASE_CONFIG:-configs/graphbert_wikitext103.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_2xt4.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baseline-vs-appnp-k8-mldr}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-16}"
RETRIEVAL_LR="${RETRIEVAL_LR:-2e-5}"
RETRIEVAL_MAX_SAMPLES="${RETRIEVAL_MAX_SAMPLES:-1250000}"
RETRIEVAL_GRADIENT_ACCUMULATION_STEPS="${RETRIEVAL_GRADIENT_ACCUMULATION_STEPS:-1}"
RETRIEVAL_FP16_FLAG="${RETRIEVAL_FP16_FLAG:---fp16}"

MLDR_SPLIT="${MLDR_SPLIT:-test}"
MLDR_QUERY_BATCH_SIZE="${MLDR_QUERY_BATCH_SIZE:-32}"
MLDR_CORPUS_BATCH_SIZE="${MLDR_CORPUS_BATCH_SIZE:-2}"
MLDR_TOP_K="${MLDR_TOP_K:-100}"
MLDR_DOCUMENT_MAX_LENGTH="${MLDR_DOCUMENT_MAX_LENGTH:-4096}"
MLDR_MAX_CORPUS_DOCUMENTS="${MLDR_MAX_CORPUS_DOCUMENTS:-}"

# Optional: point these at an already-finished baseline so this script will
# reuse it instead of training baseline again.
BASELINE_MLM_DIR="${BASELINE_MLM_DIR:-}"
BASELINE_RETRIEVAL_DIR="${BASELINE_RETRIEVAL_DIR:-}"
TRAIN_BASELINE_MLM="${TRAIN_BASELINE_MLM:-false}"

mkdir -p "${OUTPUT_ROOT}/generated_configs" "${OUTPUT_ROOT}/console_logs" "${OUTPUT_ROOT}/mldr"

make_config() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local appnp_steps="$6"
  local output_dir="${OUTPUT_ROOT}/mlm/${run_id}"
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"

  python - "${BASE_CONFIG}" "${config_path}" "${output_dir}" "${num_layers}" "${strategy}" "${normalization}" "${indices}" "${appnp_steps}" "${MAX_SEQ_LENGTH}" <<'PY'
import sys
from pathlib import Path
import yaml

base, target, output_dir, count, strategy, normalization, indices, appnp_steps, max_length = sys.argv[1:]
with Path(base).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["output_dir"] = output_dir
config["dataset"]["max_seq_length"] = int(max_length)

graph = config["graph"]
graph["num_replaced_layers"] = int(count)
graph["replacement_strategy"] = strategy
graph["layer_indices"] = [int(value) for value in indices.split(",") if value]
graph["appnp_steps"] = int(appnp_steps)
graph["renormalize_adjacency"] = normalization == "row"
graph["symmetric_normalization"] = normalization == "symmetric"

Path(target).parent.mkdir(parents=True, exist_ok=True)
with Path(target).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

latest_mlm_checkpoint() {
  local run_dir="$1"

  if [[ -f "${run_dir}/model.safetensors" || -f "${run_dir}/pytorch_model.bin" ]]; then
    echo "${run_dir}"
    return
  fi

  python - "${run_dir}" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
checkpoints = []
for path in run_dir.glob("checkpoint-*"):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if path.is_dir() and match:
        checkpoints.append((int(match.group(1)), path))

if not checkpoints:
    raise SystemExit(f"No model weights or checkpoint-* directory found under {run_dir}")

print(max(checkpoints)[1])
PY
}

has_mlm_checkpoint() {
  local run_dir="$1"

  if [[ -f "${run_dir}/model.safetensors" || -f "${run_dir}/pytorch_model.bin" ]]; then
    return 0
  fi

  python - "${run_dir}" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
for path in run_dir.glob("checkpoint-*"):
    if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

default_mlm_dir() {
  local run_id="$1"

  if [[ "${run_id}" == "baseline_longformer" && -n "${BASELINE_MLM_DIR}" ]]; then
    echo "${BASELINE_MLM_DIR}"
    return
  fi

  if [[ "${run_id}" == "baseline_longformer" && -z "${BASELINE_MLM_DIR}" ]]; then
    local legacy_dir="outputs/baseline-vs-appnp-k8/baseline_longformer"
    if [[ -d "${legacy_dir}" ]] && has_mlm_checkpoint "${legacy_dir}"; then
      echo "${legacy_dir}"
      return
    fi
  fi

  echo "${OUTPUT_ROOT}/mlm/${run_id}"
}

default_retrieval_dir() {
  local run_id="$1"

  if [[ "${run_id}" == "baseline_longformer" && -n "${BASELINE_RETRIEVAL_DIR}" ]]; then
    echo "${BASELINE_RETRIEVAL_DIR}"
    return
  fi

  echo "${OUTPUT_ROOT}/retrieval/${run_id}/single-vector-msmarco"
}

copy_tokenizer_to_checkpoint() {
  local mlm_dir="$1"
  local checkpoint="$2"

  if [[ "${checkpoint}" == "${mlm_dir}" ]]; then
    return
  fi

  python - "${mlm_dir}" "${checkpoint}" <<'PY'
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
for name in (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
):
    source_file = source / name
    target_file = target / name
    if source_file.exists() and not target_file.exists():
        shutil.copy2(source_file, target_file)
PY
}

run_mlm_train() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local appnp_steps="$6"
  local run_dir
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"
  local console_log="${OUTPUT_ROOT}/console_logs/${run_id}.mlm_train.log"

  make_config "${run_id}" "${num_layers}" "${strategy}" "${normalization}" "${indices}" "${appnp_steps}"
  run_dir="$(default_mlm_dir "${run_id}")"
  mkdir -p "${run_dir}"

  if [[ -f "${run_dir}/MLM_TRAIN_FINISHED" ]] || has_mlm_checkpoint "${run_dir}"; then
    echo "Skipping MLM training for ${run_id}; already finished."
    return
  fi

  if [[ "${run_id}" == "baseline_longformer" && "${TRAIN_BASELINE_MLM}" != "true" ]]; then
    echo "Baseline MLM checkpoint not found at ${run_dir}." >&2
    echo "Set BASELINE_MLM_DIR=/path/to/existing/baseline or TRAIN_BASELINE_MLM=true if you really want to retrain it." >&2
    return 1
  fi

  echo "Starting MLM training for ${run_id}"
  accelerate launch --config_file "${ACCELERATE_CONFIG}" scripts/train_mlm.py --config "${config_path}" \
    > "${console_log}" 2>&1
  printf '{"run_id":"%s","finished_at":"%s"}\n' "${run_id}" "$(date +%s)" > "${run_dir}/MLM_TRAIN_FINISHED"
  echo "Finished MLM training for ${run_id}"
}

run_retrieval_train() {
  local run_id="$1"
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"
  local mlm_dir
  local retrieval_dir
  local console_log="${OUTPUT_ROOT}/console_logs/${run_id}.retrieval_train.log"
  local source_model

  mlm_dir="$(default_mlm_dir "${run_id}")"
  retrieval_dir="$(default_retrieval_dir "${run_id}")"

  if [[ -f "${retrieval_dir}/RETRIEVAL_TRAIN_FINISHED" || -f "${retrieval_dir}/retrieval_model.pt" ]]; then
    echo "Skipping retrieval training for ${run_id}; already finished."
    return
  fi

  source_model="$(latest_mlm_checkpoint "${mlm_dir}")"
  copy_tokenizer_to_checkpoint "${mlm_dir}" "${source_model}"
  mkdir -p "${retrieval_dir}"

  echo "Starting MS MARCO retrieval training for ${run_id} from ${source_model}"
  python scripts/train_retrieval.py \
    --config "${config_path}" \
    --source-model "${source_model}" \
    --output-dir "${retrieval_dir}" \
    --stage msmarco \
    --architecture single \
    --batch-size "${RETRIEVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${RETRIEVAL_GRADIENT_ACCUMULATION_STEPS}" \
    --learning-rate "${RETRIEVAL_LR}" \
    --warmup-ratio 0.05 \
    --max-samples "${RETRIEVAL_MAX_SAMPLES}" \
    --gradient-checkpointing \
    ${RETRIEVAL_FP16_FLAG} \
    > "${console_log}" 2>&1
  printf '{"run_id":"%s","source_model":"%s","finished_at":"%s"}\n' "${run_id}" "${source_model}" "$(date +%s)" > "${retrieval_dir}/RETRIEVAL_TRAIN_FINISHED"
  echo "Finished MS MARCO retrieval training for ${run_id}"
}

run_mldr_eval() {
  local run_id="$1"
  local retrieval_dir
  local eval_dir="${OUTPUT_ROOT}/mldr/${run_id}/single-vector-ood"
  local console_log="${OUTPUT_ROOT}/console_logs/${run_id}.mldr_eval.log"

  retrieval_dir="$(default_retrieval_dir "${run_id}")"

  if [[ -f "${eval_dir}/MLDR_EVAL_FINISHED" ]]; then
    echo "Skipping MLDR evaluation for ${run_id}; already finished."
    return
  fi

  mkdir -p "${eval_dir}"

  local max_corpus_args=()
  if [[ -n "${MLDR_MAX_CORPUS_DOCUMENTS}" ]]; then
    max_corpus_args=(--max-corpus-documents "${MLDR_MAX_CORPUS_DOCUMENTS}")
  fi

  echo "Starting MLDR evaluation for ${run_id}"
  python scripts/evaluate_mldr.py \
    --checkpoint "${retrieval_dir}" \
    --split "${MLDR_SPLIT}" \
    --output-dir "${eval_dir}" \
    --query-batch-size "${MLDR_QUERY_BATCH_SIZE}" \
    --corpus-batch-size "${MLDR_CORPUS_BATCH_SIZE}" \
    --top-k "${MLDR_TOP_K}" \
    --document-max-length "${MLDR_DOCUMENT_MAX_LENGTH}" \
    "${max_corpus_args[@]}" \
    > "${console_log}" 2>&1
  printf '{"run_id":"%s","checkpoint":"%s","finished_at":"%s"}\n' "${run_id}" "${retrieval_dir}" "$(date +%s)" > "${eval_dir}/MLDR_EVAL_FINISHED"
  echo "Finished MLDR evaluation for ${run_id}"
}

run_one() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local appnp_steps="$6"

  run_mlm_train "${run_id}" "${num_layers}" "${strategy}" "${normalization}" "${indices}" "${appnp_steps}"
  run_retrieval_train "${run_id}"
  run_mldr_eval "${run_id}"
}

# run_id|num_appnp_adapter_layers|placement_strategy|normalization|explicit_indices|appnp_steps
RUNS=(
  "baseline_longformer|0|final|symmetric||8"
  "final_2_appnp_k8|2|final|symmetric||8"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r run_id num_layers strategy normalization indices appnp_steps <<< "${spec}"
  run_one "${run_id}" "${num_layers}" "${strategy}" "${normalization}" "${indices}" "${appnp_steps}"
done
