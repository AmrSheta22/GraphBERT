from __future__ import annotations

import math
from typing import Dict

from transformers import TrainerCallback

from graphbert.modeling import iter_graph_attention_modules


def collect_graph_stats(model) -> Dict[str, float]:
    stats = []
    unwrapped = getattr(model, "module", model)
    for module in iter_graph_attention_modules(unwrapped):
        if module.latest_graph_stats:
            stats.append(module.latest_graph_stats)

    if not stats:
        return {
            "graph_sparsity": 0.0,
            "graph_avg_degree": 0.0,
            "graph_surviving_edge_pct": 100.0,
        }

    keys = stats[0].keys()
    return {key: sum(item[key] for item in stats) / len(stats) for key in keys}


def add_perplexity(metrics: Dict[str, float], loss_key: str = "eval_loss") -> Dict[str, float]:
    if loss_key in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics[loss_key])
        except OverflowError:
            metrics["perplexity"] = float("inf")
    return metrics


class GraphStatsCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and model is not None:
            logs.update(collect_graph_stats(model))

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is not None and model is not None:
            metrics.update(collect_graph_stats(model))
            add_perplexity(metrics)
