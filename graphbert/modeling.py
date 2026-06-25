from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, LongformerForMaskedLM

from graphbert.config import GraphAttentionConfig
from graphbert.graph_attention import APPNPLongformerLayer


def selected_layer_indices(num_layers: int, graph_config: GraphAttentionConfig) -> List[int]:
    graph_config.validate(num_hidden_layers=num_layers)
    count = graph_config.num_replaced_layers
    if count == 0:
        return []

    strategy = graph_config.replacement_strategy
    if strategy == "final":
        return list(range(num_layers - count, num_layers))
    if strategy == "intermediate":
        start = (num_layers - count) // 2
        return list(range(start, start + count))
    if strategy == "first":
        return list(range(count))
    if strategy == "uniform":
        if count == 1:
            return [num_layers // 2]
        return [round(i * (num_layers - 1) / (count - 1)) for i in range(count)]
    if strategy == "explicit":
        return sorted(graph_config.layer_indices)
    raise ValueError(f"Unknown replacement strategy: {strategy}")


def add_longformer_appnp_adapters(
    model: LongformerForMaskedLM,
    graph_config: GraphAttentionConfig,
) -> List[int]:
    layers = model.longformer.encoder.layer
    indices = selected_layer_indices(len(layers), graph_config)
    for idx in indices:
        layers[idx] = APPNPLongformerLayer.from_longformer_layer(
            layers[idx],
            model.config,
            idx,
            graph_config,
        )
    model.config.graphbert = dict(graph_config.__dict__)
    model.config.graphbert_replaced_layers = indices
    model.config.graphbert_adapted_layers = indices
    return indices


# Compatibility aliases for existing scripts and saved experiment code.
replacement_layer_indices = selected_layer_indices
add_longformer_gcn_adapters = add_longformer_appnp_adapters
replace_longformer_layers = add_longformer_appnp_adapters


def iter_graph_attention_modules(model) -> Iterable[APPNPLongformerLayer]:
    for module in model.modules():
        if isinstance(module, APPNPLongformerLayer):
            yield module


def build_graph_bert_for_mlm(
    model_name_or_path: str,
    graph_config: GraphAttentionConfig,
) -> LongformerForMaskedLM:
    config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type != "longformer":
        raise ValueError(
            f"Expected a Longformer checkpoint, got model_type={config.model_type!r} "
            f"from {model_name_or_path!r}."
        )
    graph_config.validate(num_hidden_layers=config.num_hidden_layers)
    model = LongformerForMaskedLM.from_pretrained(model_name_or_path, config=config)
    add_longformer_appnp_adapters(model, graph_config)
    return model


def load_graph_bert_checkpoint(
    checkpoint: str,
    graph_config: GraphAttentionConfig,
) -> LongformerForMaskedLM:
    """Load a saved checkpoint after recreating its APPNP residual adapters."""
    checkpoint_path = Path(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint)
    if config.model_type != "longformer":
        raise ValueError(f"Expected a Longformer checkpoint, got model_type={config.model_type!r}.")
    safetensors_path = checkpoint_path / "model.safetensors"
    pytorch_path = checkpoint_path / "pytorch_model.bin"
    if safetensors_path.exists():
        state_dict = load_safetensors(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint}")

    has_appnp_adapters = any(
        ".appnp_projection." in key or key.endswith(".appnp_gate")
        for key in state_dict
    )
    has_legacy_gcn = any(".gcn." in key or key.endswith(".gcn_gate") for key in state_dict)
    if has_legacy_gcn:
        raise RuntimeError(
            "This checkpoint uses a retired GCN architecture. Start a new run from "
            "its original Longformer base checkpoint to use APPNP residual adapters."
        )

    model = LongformerForMaskedLM(config)
    effective_graph_config = graph_config
    saved_graph_config = getattr(config, "graphbert", None)
    if has_appnp_adapters and saved_graph_config:
        valid_fields = GraphAttentionConfig.__dataclass_fields__.keys()
        effective_graph_config = GraphAttentionConfig(
            **{key: value for key, value in saved_graph_config.items() if key in valid_fields}
        )
    if has_appnp_adapters:
        add_longformer_appnp_adapters(model, effective_graph_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.decoder.weight", "lm_head.decoder.bias"}
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture does not match the configured residual APPNP model. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    model.tie_weights()
    if not has_appnp_adapters:
        add_longformer_appnp_adapters(model, graph_config)
    return model
