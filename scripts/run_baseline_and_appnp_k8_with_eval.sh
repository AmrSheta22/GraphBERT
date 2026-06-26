#!/usr/bin/env bash
set -Eeuo pipefail

BASE_CONFIG="${BASE_CONFIG:-configs/graphbert_wikitext103.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_2xt4.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baseline-vs-appnp-k8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

mkdir -p "${OUTPUT_ROOT}/generated_configs" "${OUTPUT_ROOT}/console_logs"

make_config() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local appnp_steps="$6"
  local output_dir="${OUTPUT_ROOT}/${run_id}"
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

latest_checkpoint() {
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

run_train() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local appnp_steps="$6"
  local run_dir="${OUTPUT_ROOT}/${run_id}"
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"
  local console_log="${OUTPUT_ROOT}/console_logs/${run_id}.train.log"

  make_config "${run_id}" "${num_layers}" "${strategy}" "${normalization}" "${indices}" "${appnp_steps}"
  mkdir -p "${run_dir}"

  if [[ -f "${run_dir}/TRAIN_FINISHED" ]]; then
    echo "Skipping training for ${run_id}; already finished."
    return
  fi

  echo "Starting training for ${run_id}"
  accelerate launch --config_file "${ACCELERATE_CONFIG}" scripts/train_mlm.py --config "${config_path}" \
    > "${console_log}" 2>&1
  printf '{"run_id":"%s","finished_at":"%s"}\n' "${run_id}" "$(date +%s)" > "${run_dir}/TRAIN_FINISHED"
  echo "Finished training for ${run_id}"
}

run_eval() {
  local run_id="$1"
  local run_dir="${OUTPUT_ROOT}/${run_id}"
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"
  local console_log="${OUTPUT_ROOT}/console_logs/${run_id}.eval.log"
  local checkpoint

  if [[ -f "${run_dir}/EVAL_FINISHED" ]]; then
    echo "Skipping evaluation for ${run_id}; already finished."
    return
  fi

  checkpoint="$(latest_checkpoint "${run_dir}")"
  echo "Starting evaluation for ${run_id} from ${checkpoint}"
  python scripts/evaluate_mlm.py --config "${config_path}" --checkpoint "${checkpoint}" \
    > "${console_log}" 2>&1
  printf '{"run_id":"%s","checkpoint":"%s","finished_at":"%s"}\n' "${run_id}" "${checkpoint}" "$(date +%s)" > "${run_dir}/EVAL_FINISHED"
  echo "Finished evaluation for ${run_id}"
}

run_one() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local appnp_steps="$6"

  run_train "${run_id}" "${num_layers}" "${strategy}" "${normalization}" "${indices}" "${appnp_steps}"
  run_eval "${run_id}"
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
