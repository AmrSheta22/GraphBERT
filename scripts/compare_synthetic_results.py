from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare APPNP and Longformer synthetic retrieval results.")
    parser.add_argument("--comparison-root", required=True)
    return parser.parse_args()


def load_run(root: Path, name: str):
    metrics_path = root / name / "results" / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    results = metrics.get("results", [])
    if not results:
        raise ValueError(f"No per-length results found in {metrics_path}")
    return {
        "checkpoint": metrics["checkpoint"],
        "metrics_file": str(metrics_path),
        "mean_ndcg_at_10": sum(row["ndcg_at_10"] for row in results) / len(results),
        "mean_recall_at_1": sum(row["recall_at_1"] for row in results) / len(results),
        "results": results,
    }


def main():
    args = parse_args()
    root = Path(args.comparison_root)
    runs = {
        name: run
        for name in ("appnp", "longformer")
        if (run := load_run(root, name)) is not None
    }
    if not runs:
        raise SystemExit(f"No metrics found below {root}")

    best_name = max(runs, key=lambda name: runs[name]["mean_ndcg_at_10"])
    lengths = sorted(
        {
            row["document_max_length"]
            for run in runs.values()
            for row in run["results"]
        }
    )
    winners_by_length = {}
    for length in lengths:
        candidates = {}
        for name, run in runs.items():
            row = next(
                (item for item in run["results"] if item["document_max_length"] == length),
                None,
            )
            if row is not None:
                candidates[name] = row["ndcg_at_10"]
        winner = max(candidates, key=candidates.get)
        winners_by_length[str(length)] = {
            "winner": winner,
            "ndcg_at_10": candidates[winner],
            "all_scores": candidates,
        }

    comparison = {
        "selection_metric": "mean_ndcg_at_10_across_configured_context_lengths",
        "status": "complete" if len(runs) == 2 else "provisional_waiting_for_other_model",
        "best_model": best_name,
        "best_checkpoint": runs[best_name]["checkpoint"],
        "runs": runs,
        "winners_by_length": winners_by_length,
    }
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "comparison.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
