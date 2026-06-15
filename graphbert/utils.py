from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import set_seed

from graphbert.config import ExperimentConfig, load_experiment_config


def parse_config_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation.")
    parser.add_argument("--num-replaced-layers", type=int, default=None)
    parser.add_argument("--sparsification", choices=["dense", "threshold", "topk"], default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def load_config_with_overrides(args) -> ExperimentConfig:
    config = load_experiment_config(args.config)
    if args.num_replaced_layers is not None:
        config.graph.num_replaced_layers = args.num_replaced_layers
    if args.sparsification is not None:
        config.graph.sparsification = args.sparsification
    if args.threshold is not None:
        config.graph.threshold = args.threshold
    if args.top_k is not None:
        config.graph.top_k = args.top_k
    return config


def prepare_reproducibility(seed: int) -> None:
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_experiment_config(config: ExperimentConfig, output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)


def compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: float(value) if isinstance(value, (np.floating, np.integer)) else value for key, value in metrics.items()}
