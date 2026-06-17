from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphbert.config import ExperimentConfig, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a greedy GraphBERT search over replacement depth, sparsification, and normalization."
    )
    parser.add_argument("--config", required=True, help="Base YAML config.")
    parser.add_argument("--output-root", default="outputs/greedy-search", help="Directory for all generated runs.")
    parser.add_argument("--log-file", default=None, help="JSONL summary log. Defaults under --output-root.")
    parser.add_argument("--launcher", choices=["python", "accelerate"], default="accelerate")
    parser.add_argument("--num-processes", type=int, default=None, help="Passed to accelerate launch when set.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and log planned runs without training.")
    parser.add_argument("--skip-baseline", action="store_true", help="Do not run the num_replaced_layers=0 baseline.")
    parser.add_argument("--layer-candidates", default="1,2,4", help="Comma-separated replacement layer counts.")
    parser.add_argument("--top-k-candidates", default="8,16,32", help="Comma-separated k values for top-k candidates.")
    parser.add_argument("--threshold-candidates", default="0.005,0.01,0.02", help="Comma-separated thresholds.")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def slugify_run(spec: Dict[str, Any]) -> str:
    parts = [spec["stage"], spec["name"]]
    graph = spec["config"]["graph"]
    parts.append(f"layers{graph['num_replaced_layers']}")
    if graph["num_replaced_layers"] > 0:
        parts.append(graph["sparsification"])
        if graph["sparsification"] == "topk":
            parts.append(f"k{graph['top_k']}")
        if graph["sparsification"] == "threshold":
            parts.append(f"thr{graph['threshold']}".replace(".", "p"))
        if graph["symmetric_normalization"]:
            parts.append("symnorm")
        elif graph["renormalize_adjacency"]:
            parts.append("rownorm")
        else:
            parts.append("nonorm")
        if graph["add_self_loops"]:
            parts.append("selfloops")
    return "_".join(str(part) for part in parts)


def configure_base(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    config = deepcopy(config)
    config.dataset.max_seq_length = args.max_seq_length
    config.training.per_device_train_batch_size = args.train_batch_size
    config.training.per_device_eval_batch_size = args.eval_batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.fp16 = args.fp16
    config.training.bf16 = args.bf16
    return config


def make_spec(stage: str, name: str, config: ExperimentConfig) -> Dict[str, Any]:
    return {"stage": stage, "name": name, "config": asdict(config)}


def write_yaml_config(spec: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec["config"], handle, sort_keys=False)


def build_command(config_path: Path, args: argparse.Namespace) -> List[str]:
    if args.launcher == "accelerate":
        command = ["accelerate", "launch"]
        if args.num_processes is not None:
            command.extend(["--num_processes", str(args.num_processes)])
        command.extend(["scripts/train_mlm.py", "--config", str(config_path)])
        return command
    return [sys.executable, "scripts/train_mlm.py", "--config", str(config_path)]


def read_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for filename in ("eval_results.json", "train_results.json", "all_results.json"):
        path = run_dir / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                metrics.update(json.load(handle))
    return metrics


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_experiment(spec: Dict[str, Any], output_root: Path, args: argparse.Namespace, summary_log: Path) -> Dict[str, Any]:
    run_id = slugify_run(spec)
    run_dir = output_root / run_id
    config_path = output_root / "generated_configs" / f"{run_id}.yaml"
    console_log = output_root / "console_logs" / f"{run_id}.log"

    spec["config"]["output_dir"] = str(run_dir)
    write_yaml_config(spec, config_path)
    command = build_command(config_path, args)

    start_payload = {
        "event": "start",
        "run_id": run_id,
        "stage": spec["stage"],
        "name": spec["name"],
        "command": command,
        "config_path": str(config_path),
        "output_dir": str(run_dir),
        "console_log": str(console_log),
        "spec": spec["config"],
        "timestamp": time.time(),
    }
    append_jsonl(summary_log, start_payload)

    if args.dry_run:
        end_payload = {**start_payload, "event": "dry_run", "returncode": None, "metrics": {}}
        append_jsonl(summary_log, end_payload)
        return end_payload

    console_log.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with console_log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    metrics = read_metrics(run_dir)
    end_payload = {
        "event": "end",
        "run_id": run_id,
        "stage": spec["stage"],
        "name": spec["name"],
        "returncode": process.returncode,
        "duration_seconds": time.time() - started_at,
        "config_path": str(config_path),
        "output_dir": str(run_dir),
        "console_log": str(console_log),
        "spec": spec["config"],
        "metrics": metrics,
        "timestamp": time.time(),
    }
    append_jsonl(summary_log, end_payload)
    if process.returncode != 0:
        raise RuntimeError(f"Run {run_id} failed. See {console_log}")
    return end_payload


def best_by_eval_loss(results: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    completed = [item for item in results if item.get("returncode") == 0 and "eval_loss" in item.get("metrics", {})]
    if not completed:
        return None
    return min(completed, key=lambda item: item["metrics"]["eval_loss"])


def set_graph(config: ExperimentConfig, **kwargs) -> ExperimentConfig:
    config = deepcopy(config)
    for key, value in kwargs.items():
        setattr(config.graph, key, value)
    return config


def layer_stage(base: ExperimentConfig, layer_candidates: List[int]) -> List[Dict[str, Any]]:
    specs = []
    for layers in layer_candidates:
        config = set_graph(
            base,
            num_replaced_layers=layers,
            sparsification="topk",
            top_k=16,
            threshold=0.01,
            renormalize_adjacency=True,
            symmetric_normalization=False,
            add_self_loops=False,
        )
        specs.append(make_spec("layers", f"topk_rownorm_layers_{layers}", config))
    return specs


def sparsification_stage(base: ExperimentConfig, top_ks: List[int], thresholds: List[float]) -> List[Dict[str, Any]]:
    specs = [
        make_spec("sparsification", "dense_rownorm", set_graph(base, sparsification="dense")),
    ]
    for top_k in top_ks:
        specs.append(make_spec("sparsification", f"topk_{top_k}", set_graph(base, sparsification="topk", top_k=top_k)))
    for threshold in thresholds:
        specs.append(
            make_spec(
                "sparsification",
                f"threshold_{threshold}",
                set_graph(base, sparsification="threshold", threshold=threshold),
            )
        )
    return specs


def normalization_stage(base: ExperimentConfig) -> List[Dict[str, Any]]:
    variants = [
        ("no_norm", False, False, False),
        ("row_norm", True, False, False),
        ("sym_norm", False, True, False),
        ("row_norm_selfloops", True, False, True),
        ("sym_norm_selfloops", False, True, True),
    ]
    return [
        make_spec(
            "normalization",
            name,
            set_graph(
                base,
                renormalize_adjacency=row_norm,
                symmetric_normalization=sym_norm,
                add_self_loops=self_loops,
            ),
        )
        for name, row_norm, sym_norm, self_loops in variants
    ]


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    summary_log = Path(args.log_file) if args.log_file else output_root / "greedy_search.jsonl"
    base_config = configure_base(load_experiment_config(args.config), args)
    layer_candidates = parse_ints(args.layer_candidates)
    top_ks = parse_ints(args.top_k_candidates)
    thresholds = parse_floats(args.threshold_candidates)

    all_results: List[Dict[str, Any]] = []

    if args.dry_run:
        if not args.skip_baseline:
            baseline = set_graph(base_config, num_replaced_layers=0)
            all_results.append(
                run_experiment(make_spec("baseline", "bert_large_uncased", baseline), output_root, args, summary_log)
            )
        layer_specs = layer_stage(base_config, layer_candidates)
        for spec in layer_specs:
            all_results.append(run_experiment(spec, output_root, args, summary_log))
        dry_greedy = load_experiment_config(all_results[-1]["config_path"]) if layer_specs else base_config
        sparsification_specs = sparsification_stage(dry_greedy, top_ks, thresholds)
        for spec in sparsification_specs:
            all_results.append(run_experiment(spec, output_root, args, summary_log))
        dry_greedy = load_experiment_config(all_results[-1]["config_path"]) if sparsification_specs else dry_greedy
        for spec in normalization_stage(dry_greedy):
            all_results.append(run_experiment(spec, output_root, args, summary_log))
        append_jsonl(
            summary_log,
            {
                "event": "summary",
                "dry_run": True,
                "planned_runs": len(all_results),
                "log_file": str(summary_log),
                "timestamp": time.time(),
            },
        )
        return

    if not args.skip_baseline:
        baseline = set_graph(base_config, num_replaced_layers=0)
        all_results.append(run_experiment(make_spec("baseline", "bert_large_uncased", baseline), output_root, args, summary_log))

    layer_results = []
    for spec in layer_stage(base_config, layer_candidates):
        layer_results.append(run_experiment(spec, output_root, args, summary_log))
    all_results.extend(layer_results)
    best_layers = best_by_eval_loss(layer_results)
    if best_layers is None:
        raise RuntimeError("No successful layer-stage result with eval_loss.")

    greedy_config = load_experiment_config(best_layers["config_path"])

    sparsification_results = []
    for spec in sparsification_stage(greedy_config, top_ks, thresholds):
        sparsification_results.append(run_experiment(spec, output_root, args, summary_log))
    all_results.extend(sparsification_results)
    best_sparsification = best_by_eval_loss(sparsification_results)
    if best_sparsification is None:
        raise RuntimeError("No successful sparsification-stage result with eval_loss.")

    greedy_config = load_experiment_config(best_sparsification["config_path"])

    normalization_results = []
    for spec in normalization_stage(greedy_config):
        normalization_results.append(run_experiment(spec, output_root, args, summary_log))
    all_results.extend(normalization_results)
    best_final = best_by_eval_loss(normalization_results) or best_by_eval_loss(all_results)

    append_jsonl(
        summary_log,
        {
            "event": "summary",
            "best_run_id": best_final["run_id"] if best_final else None,
            "best_eval_loss": best_final["metrics"].get("eval_loss") if best_final else None,
            "best_output_dir": best_final.get("output_dir") if best_final else None,
            "best_spec": best_final.get("spec") if best_final else None,
            "timestamp": time.time(),
        },
    )


if __name__ == "__main__":
    main()
