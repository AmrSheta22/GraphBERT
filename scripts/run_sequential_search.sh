#!/usr/bin/env bash
set -Eeuo pipefail

BASE_CONFIG="${BASE_CONFIG:-configs/graphbert_wikitext103.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_2xt4.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sequential-search}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

mkdir -p "${OUTPUT_ROOT}/generated_configs" "${OUTPUT_ROOT}/console_logs"

make_config() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local output_dir="${OUTPUT_ROOT}/${run_id}"
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"

  python - "${BASE_CONFIG}" "${config_path}" "${output_dir}" "${num_layers}" "${strategy}" "${normalization}" "${indices}" "${MAX_SEQ_LENGTH}" <<'PY'
import sys
from pathlib import Path
import yaml

base, target, output_dir, count, strategy, normalization, indices, max_length = sys.argv[1:]
with Path(base).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["output_dir"] = output_dir
config["dataset"]["max_seq_length"] = int(max_length)
graph = config["graph"]
graph["num_replaced_layers"] = int(count)
graph["replacement_strategy"] = strategy
graph["layer_indices"] = [int(value) for value in indices.split(",") if value]
graph["renormalize_adjacency"] = normalization == "row"
graph["symmetric_normalization"] = normalization == "symmetric"

Path(target).parent.mkdir(parents=True, exist_ok=True)
with Path(target).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

run_one() {
  local run_id="$1"
  local num_layers="$2"
  local strategy="$3"
  local normalization="$4"
  local indices="$5"
  local run_dir="${OUTPUT_ROOT}/${run_id}"
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"
  local console_log="${OUTPUT_ROOT}/console_logs/${run_id}.log"

  if [[ -f "${run_dir}/FINISHED" ]]; then
    echo "Skipping ${run_id}; already finished."
    return
  fi

  mkdir -p "${run_dir}"
  make_config "${run_id}" "${num_layers}" "${strategy}" "${normalization}" "${indices}"
  echo "Starting ${run_id}"
  accelerate launch --config_file "${ACCELERATE_CONFIG}" scripts/train_mlm.py --config "${config_path}" \
    > "${console_log}" 2>&1
  printf '{"run_id":"%s","finished_at":"%s"}\n' "${run_id}" "$(date +%s)" > "${run_dir}/FINISHED"
  echo "Finished ${run_id}"
}

# run_id|num_replaced_layers|replacement_strategy|normalization|explicit_indices
RUNS=(
  "baseline_longformer|0|final|row|"
  "final_2_gcn|2|final|row|"
  "intermediate_2_gcn|2|intermediate|row|"
  "uniform_3_gcn|3|uniform|row|"
  "explicit_5_8_gcn|2|explicit|row|5,8"
  "final_4_gcn_symmetric|4|final|symmetric|"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r run_id num_layers strategy normalization indices <<< "${spec}"
  run_one "${run_id}" "${num_layers}" "${strategy}" "${normalization}" "${indices}"
done
