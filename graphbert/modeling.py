from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, BertForMaskedLM

from graphbert.config import GraphAttentionConfig
from graphbert.graph_attention import GraphBertSelfAttention


def replacement_layer_indices(num_layers: int, graph_config: GraphAttentionConfig) -> List[int]:
    graph_config.validate(num_hidden_layers=num_layers)
    n = graph_config.num_replaced_layers
    if n == 0:
        return []
    if graph_config.replace_final_layers:
        return list(range(num_layers - n, num_layers))
    return list(range(n))


def replace_bert_attention_layers(model: BertForMaskedLM, graph_config: GraphAttentionConfig) -> List[int]:
    layers = model.bert.encoder.layer
    indices = replacement_layer_indices(len(layers), graph_config)
    for idx in indices:
        bert_self_attention = layers[idx].attention.self
        layers[idx].attention.self = GraphBertSelfAttention.from_bert_self_attention(
            bert_self_attention,
            model.config,
            graph_config,
        )
    model.config.graphbert = dict(graph_config.__dict__)
    model.config.graphbert_replaced_layers = indices
    return indices


def iter_graph_attention_modules(model) -> Iterable[GraphBertSelfAttention]:
    for module in model.modules():
        if isinstance(module, GraphBertSelfAttention):
            yield module


def build_graph_bert_for_mlm(model_name_or_path: str, graph_config: GraphAttentionConfig) -> BertForMaskedLM:
    config = AutoConfig.from_pretrained(model_name_or_path)
    graph_config.validate(num_hidden_layers=config.num_hidden_layers)
    model = BertForMaskedLM.from_pretrained(model_name_or_path, config=config)
    replace_bert_attention_layers(model, graph_config)
    return model


def load_graph_bert_checkpoint(checkpoint: str, graph_config: GraphAttentionConfig) -> BertForMaskedLM:
    """Load a saved GraphBERT checkpoint after recreating swapped attention modules."""
    model = BertForMaskedLM.from_pretrained(checkpoint)
    replace_bert_attention_layers(model, graph_config)

    checkpoint_path = Path(checkpoint)
    safetensors_path = checkpoint_path / "model.safetensors"
    pytorch_path = checkpoint_path / "pytorch_model.bin"
    if safetensors_path.exists():
        state_dict = load_safetensors(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint}")

    model.load_state_dict(state_dict, strict=False)
    return model
