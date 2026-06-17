#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential, resumable experiment runner for Kaggle-like 2xT4 environments.
#
# It runs one experiment at a time. After each successful run it writes a FINISHED
# marker, commits lightweight artifacts, and pushes to the current Git remote.
# Large model weights are intentionally excluded unless INCLUDE_CHECKPOINTS=1.

BASE_CONFIG="${BASE_CONFIG:-configs/graphbert_wikitext103.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_2xt4.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sequential-search}"
SUMMARY_LOG="${SUMMARY_LOG:-${OUTPUT_ROOT}/sequential_search.jsonl}"
COMMIT_RESULTS="${COMMIT_RESULTS:-1}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
FP16="${FP16:-true}"
BF16="${BF16:-false}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}/generated_configs" "${OUTPUT_ROOT}/console_logs"

json_escape() {
  python -c 'import json, sys; print(json.dumps(sys.stdin.read()))'
}

append_event() {
  local payload="$1"
  printf '%s\n' "${payload}" >> "${SUMMARY_LOG}"
}

make_config() {
  local run_id="$1"
  local output_dir="$2"
  local num_layers="$3"
  local sparsification="$4"
  local top_k="$5"
  local threshold="$6"
  local normalization="$7"
  local self_loops="$8"
  local config_path="$9"

  python - "$BASE_CONFIG" "$config_path" "$output_dir" "$num_layers" "$sparsification" "$top_k" "$threshold" "$normalization" "$self_loops" "$MAX_SEQ_LENGTH" "$TRAIN_BATCH_SIZE" "$EVAL_BATCH_SIZE" "$GRAD_ACCUM_STEPS" "$FP16" "$BF16" <<'PY'
import sys
from pathlib import Path

import yaml

(
    base_config,
    config_path,
    output_dir,
    num_layers,
    sparsification,
    top_k,
    threshold,
    normalization,
    self_loops,
    max_seq_length,
    train_batch_size,
    eval_batch_size,
    grad_accum_steps,
    fp16,
    bf16,
) = sys.argv[1:]

with Path(base_config).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["output_dir"] = output_dir
config["dataset"]["max_seq_length"] = int(max_seq_length)
num_layers_int = int(num_layers)
train_batch_size_int = int(train_batch_size)
eval_batch_size_int = int(eval_batch_size)
grad_accum_steps_int = int(grad_accum_steps)
memory_safe = num_layers_int >= 4
if memory_safe:
    reduction_factor = 4
    train_batch_size_int = max(1, train_batch_size_int // reduction_factor)
    eval_batch_size_int = max(1, eval_batch_size_int // reduction_factor)
    grad_accum_steps_int = grad_accum_steps_int * reduction_factor

config["training"]["per_device_train_batch_size"] = train_batch_size_int
config["training"]["per_device_eval_batch_size"] = eval_batch_size_int
config["training"]["gradient_accumulation_steps"] = grad_accum_steps_int
config["training"]["fp16"] = fp16.lower() == "true"
config["training"]["bf16"] = bf16.lower() == "true"
config["training"]["gradient_checkpointing"] = memory_safe
config["training"]["overwrite_output_dir"] = False
config["training"]["save_strategy"] = "steps"
config["training"]["save_only_model"] = False
config["training"]["save_total_limit"] = 1

graph = config["graph"]
graph["num_replaced_layers"] = num_layers_int
graph["sparsification"] = sparsification
graph["top_k"] = int(top_k)
graph["threshold"] = float(threshold)
graph["add_self_loops"] = self_loops.lower() == "true"

if normalization == "none":
    graph["renormalize_adjacency"] = False
    graph["symmetric_normalization"] = False
elif normalization == "row":
    graph["renormalize_adjacency"] = True
    graph["symmetric_normalization"] = False
elif normalization == "symmetric":
    graph["renormalize_adjacency"] = False
    graph["symmetric_normalization"] = True
else:
    raise ValueError(f"Unknown normalization: {normalization}")

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
with Path(config_path).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
}

commit_and_push() {
  local run_id="$1"
  local run_dir="$2"
  local config_path="$3"
  local console_log="$4"

  if [[ "${COMMIT_RESULTS}" != "1" ]]; then
    return 0
  fi

  git add scripts/run_sequential_search.sh configs/accelerate_2xt4.yaml configs/graphbert_wikitext103.yaml README.md .gitignore || true
  git add -f "${config_path}" "${console_log}" "${SUMMARY_LOG}" "${run_dir}/FINISHED" || true

  for metrics_file in "${run_dir}"/*_results.json "${run_dir}/trainer_state.json" "${run_dir}/resolved_config.json"; do
    if [[ -f "${metrics_file}" ]]; then
      git add -f "${metrics_file}" || true
    fi
  done

  if [[ "${INCLUDE_CHECKPOINTS}" == "1" ]]; then
    git add -f "${run_dir}" || true
  fi

  if git diff --cached --quiet; then
    echo "No changes to commit for ${run_id}."
  else
    git commit -m "Add GraphBERT run ${run_id}"
  fi

  if [[ "${PUSH_RESULTS}" == "1" ]]; then
    git push
  fi
}

run_one() {
  local run_id="$1"
  local num_layers="$2"
  local sparsification="$3"
  local top_k="$4"
  local threshold="$5"
  local normalization="$6"
  local self_loops="$7"

  local run_dir="${OUTPUT_ROOT}/${run_id}"
  local config_path="${OUTPUT_ROOT}/generated_configs/${run_id}.yaml"
  local console_log="${OUTPUT_ROOT}/console_logs/${run_id}.log"
  local finished="${run_dir}/FINISHED"

  if [[ -f "${finished}" ]]; then
    echo "Skipping ${run_id}; found ${finished}."
    return 0
  fi

  mkdir -p "${run_dir}"
  make_config "${run_id}" "${run_dir}" "${num_layers}" "${sparsification}" "${top_k}" "${threshold}" "${normalization}" "${self_loops}" "${config_path}"

  local command=(accelerate launch --config_file "${ACCELERATE_CONFIG}" scripts/train_mlm.py --config "${config_path}")
  echo "Starting ${run_id}"
  echo "Config: ${config_path}"
  echo "Log: ${console_log}"

  append_event "{\"event\":\"start\",\"run_id\":\"${run_id}\",\"output_dir\":\"${run_dir}\",\"config_path\":\"${config_path}\",\"console_log\":\"${console_log}\",\"num_replaced_layers\":${num_layers},\"sparsification\":\"${sparsification}\",\"top_k\":${top_k},\"threshold\":${threshold},\"normalization\":\"${normalization}\",\"self_loops\":${self_loops},\"max_seq_length\":${MAX_SEQ_LENGTH},\"train_batch_size\":${TRAIN_BATCH_SIZE},\"eval_batch_size\":${EVAL_BATCH_SIZE},\"gradient_accumulation_steps\":${GRAD_ACCUM_STEPS},\"fp16\":${FP16},\"bf16\":${BF16},\"timestamp\":$(date +%s)}"

  set +e
  "${command[@]}" > "${console_log}" 2>&1
  local status=$?
  set -e

  if [[ "${status}" -ne 0 ]]; then
    append_event "{\"event\":\"failed\",\"run_id\":\"${run_id}\",\"returncode\":${status},\"console_log\":\"${console_log}\",\"timestamp\":$(date +%s)}"
    echo "Run ${run_id} failed with status ${status}. See ${console_log}."
    exit "${status}"
  fi

  python - "${run_dir}" "${run_id}" "${SUMMARY_LOG}" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
run_id = sys.argv[2]
summary_log = Path(sys.argv[3])

metrics = {}
for name in ("train_results.json", "eval_results.json", "all_results.json"):
    path = run_dir / name
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            metrics.update(json.load(handle))

checkpoints = sorted(run_dir.glob("checkpoint-*"))
last_checkpoint = str(checkpoints[-1]) if checkpoints else None

(run_dir / "FINISHED").write_text(
    json.dumps(
        {
            "run_id": run_id,
            "finished_at": time.time(),
            "last_checkpoint": last_checkpoint,
            "metrics": metrics,
        },
        indent=2,
    ),
    encoding="utf-8",
)

with summary_log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "event": "finished",
        "run_id": run_id,
        "output_dir": str(run_dir),
        "last_checkpoint": last_checkpoint,
        "metrics": metrics,
        "timestamp": time.time(),
    }, sort_keys=True) + "\n")
PY

  commit_and_push "${run_id}" "${run_dir}" "${config_path}" "${console_log}"
  echo "Finished ${run_id}"
}

# Format:
# run_id|num_layers|sparsification|top_k|threshold|normalization|self_loops
#
# This is intentionally sequential and explicit. Add/remove lines here to
# control exactly how many one-hour runs you spend.
RUNS=(
  "baseline_bert_large|0|dense|16|0.01|row|false"

  "layers2_dense_row|2|dense|16|0.01|row|false"
  "layers2_topk8_row|2|topk|8|0.01|row|false"
  "layers2_topk32_row|2|topk|32|0.01|row|false"
  "layers2_threshold005_row|2|threshold|16|0.005|row|false"
  "layers2_threshold010_row|2|threshold|16|0.01|row|false"
  "layers2_threshold020_row|2|threshold|16|0.02|row|false"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r run_id num_layers sparsification top_k threshold normalization self_loops <<< "${spec}"
  run_one "${run_id}" "${num_layers}" "${sparsification}" "${top_k}" "${threshold}" "${normalization}" "${self_loops}"
done

append_event "{\"event\":\"all_done\",\"timestamp\":$(date +%s)}"
echo "All sequential runs complete."
