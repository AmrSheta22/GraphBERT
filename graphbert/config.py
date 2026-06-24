from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str = "Salesforce/wikitext"
    config_name: Optional[str] = "wikitext-103-raw-v1"
    text_column: str = "text"
    validation_split_percentage: int = 5
    max_seq_length: int = 4096
    preprocessing_num_workers: int = 4
    line_by_line: bool = False
    global_attention_on_cls: bool = True


@dataclass
class GraphAttentionConfig:
    num_replaced_layers: int = 0
    replacement_strategy: str = "final"
    layer_indices: List[int] = field(default_factory=list)
    renormalize_adjacency: bool = True
    symmetric_normalization: bool = False
    add_self_loops: bool = True
    gcn_weight_init: str = "identity"
    gcn_bias: bool = True
    gcn_activation: str = "gelu"
    gcn_dropout: float = 0.1

    def validate(self, num_hidden_layers: Optional[int] = None) -> None:
        if self.replacement_strategy not in {"final", "intermediate", "first", "uniform", "explicit"}:
            raise ValueError("graph.replacement_strategy must be final, intermediate, first, uniform, or explicit")
        if self.num_replaced_layers < 0:
            raise ValueError("graph.num_replaced_layers must be non-negative")
        if num_hidden_layers is not None and self.num_replaced_layers > num_hidden_layers:
            raise ValueError(
                f"Cannot replace {self.num_replaced_layers} layers in a model with "
                f"{num_hidden_layers} hidden layers."
            )
        if self.replacement_strategy == "explicit":
            if len(self.layer_indices) != self.num_replaced_layers:
                raise ValueError("graph.layer_indices must contain num_replaced_layers entries")
            if len(set(self.layer_indices)) != len(self.layer_indices):
                raise ValueError("graph.layer_indices must not contain duplicates")
            if num_hidden_layers is not None and any(
                index < 0 or index >= num_hidden_layers for index in self.layer_indices
            ):
                raise ValueError("graph.layer_indices contains an out-of-range layer index")
        if self.gcn_weight_init not in {"identity", "xavier"}:
            raise ValueError("graph.gcn_weight_init must be identity or xavier")
        if self.gcn_activation not in {"none", "gelu", "relu", "silu"}:
            raise ValueError("graph.gcn_activation must be none, gelu, relu, or silu")
        if not 0.0 <= self.gcn_dropout < 1.0:
            raise ValueError("graph.gcn_dropout must be in [0, 1)")
        if self.renormalize_adjacency and self.symmetric_normalization:
            raise ValueError("Choose row or symmetric normalization, not both")


@dataclass
class TrainingConfig:
    do_train: bool = True
    do_eval: bool = True
    overwrite_output_dir: bool = False
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
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
    save_strategy: str = "steps"
    save_only_model: bool = False
    save_total_limit: int = 1
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    report_to: str = "tensorboard"
    mlm_probability: float = 0.15


@dataclass
class ExperimentConfig:
    model_name_or_path: str = "allenai/longformer-base-4096"
    output_dir: str = "outputs/longformer-gcn"
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
