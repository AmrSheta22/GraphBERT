from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str = "Salesforce/wikitext"
    config_name: Optional[str] = "wikitext-103-raw-v1"
    text_column: str = "text"
    validation_split_percentage: int = 5
    max_seq_length: int = 128
    preprocessing_num_workers: int = 4
    line_by_line: bool = False


@dataclass
class GraphAttentionConfig:
    num_replaced_layers: int = 0
    replace_final_layers: bool = True
    sparsification: str = "dense"
    threshold: float = 0.0
    top_k: int = 16
    renormalize_adjacency: bool = True
    symmetric_normalization: bool = False
    add_self_loops: bool = False
    gcn_weight_init: str = "identity"

    def validate(self, num_hidden_layers: Optional[int] = None) -> None:
        if self.sparsification not in {"dense", "threshold", "topk"}:
            raise ValueError("graph.sparsification must be one of: dense, threshold, topk")
        if self.num_replaced_layers < 0:
            raise ValueError("graph.num_replaced_layers must be non-negative")
        if num_hidden_layers is not None and self.num_replaced_layers > num_hidden_layers:
            raise ValueError(
                f"Cannot replace {self.num_replaced_layers} layers in a model with "
                f"{num_hidden_layers} hidden layers."
            )
        if self.top_k <= 0:
            raise ValueError("graph.top_k must be positive")
        if self.threshold < 0:
            raise ValueError("graph.threshold must be non-negative")


@dataclass
class TrainingConfig:
    do_train: bool = True
    do_eval: bool = True
    overwrite_output_dir: bool = False
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_steps: int = 1000
    num_train_epochs: Optional[float] = None
    warmup_ratio: float = 0.06
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 250
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 2
    report_to: str = "tensorboard"
    mlm_probability: float = 0.15


@dataclass
class ExperimentConfig:
    model_name_or_path: str = "bert-large-uncased"
    output_dir: str = "outputs/graphbert"
    seed: int = 1337
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    graph: GraphAttentionConfig = field(default_factory=GraphAttentionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _dataclass_from_dict(cls, values: Dict[str, Any]):
    field_names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in values.items() if key in field_names})


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = _dataclass_from_dict(DatasetConfig, raw.get("dataset", {}))
    graph = _dataclass_from_dict(GraphAttentionConfig, raw.get("graph", {}))
    training = _dataclass_from_dict(TrainingConfig, raw.get("training", {}))
    top_level = {key: raw[key] for key in ("model_name_or_path", "output_dir", "seed") if key in raw}
    return ExperimentConfig(dataset=dataset, graph=graph, training=training, **top_level)
