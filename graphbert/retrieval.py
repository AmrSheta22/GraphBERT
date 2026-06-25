from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from graphbert.config import GraphAttentionConfig
from graphbert.modeling import build_graph_bert_for_mlm, load_graph_bert_checkpoint


@dataclass
class RetrievalConfig:
    source_model: str
    graph_config: dict
    pooling: str = "mean"
    projection_dim: int = 0
    normalize: bool = True
    query_max_length: int = 64
    document_max_length: int = 4096


class LongContextRetriever(nn.Module):
    """Single-vector or ColBERT-style retriever over Longformer-APPNP."""

    def __init__(self, encoder, config: RetrievalConfig):
        super().__init__()
        self.encoder = encoder
        self.retrieval_config = config
        hidden_size = encoder.config.hidden_size
        self.projection = (
            nn.Linear(hidden_size, config.projection_dim, bias=False)
            if config.projection_dim > 0
            else nn.Identity()
        )

    @classmethod
    def from_source(
        cls,
        source_model: str,
        graph_config: GraphAttentionConfig,
        pooling: str = "mean",
        projection_dim: int = 0,
        normalize: bool = True,
        query_max_length: int = 64,
        document_max_length: int = 4096,
    ) -> "LongContextRetriever":
        path = Path(source_model)
        if path.is_dir() and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()):
            mlm_model = load_graph_bert_checkpoint(source_model, graph_config)
        else:
            mlm_model = build_graph_bert_for_mlm(source_model, graph_config)
        config = RetrievalConfig(
            source_model=source_model,
            graph_config=asdict(graph_config),
            pooling=pooling,
            projection_dim=projection_dim,
            normalize=normalize,
            query_max_length=query_max_length,
            document_max_length=document_max_length,
        )
        return cls(mlm_model.longformer, config)

    @classmethod
    def load(cls, checkpoint: str, source_override: Optional[str] = None) -> "LongContextRetriever":
        checkpoint_path = Path(checkpoint)
        with (checkpoint_path / "retrieval_config.json").open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if source_override is not None:
            raw["source_model"] = source_override
        config = RetrievalConfig(**raw)
        graph_config = GraphAttentionConfig(**config.graph_config)
        model = cls.from_source(
            config.source_model,
            graph_config,
            pooling=config.pooling,
            projection_dim=config.projection_dim,
            normalize=config.normalize,
            query_max_length=config.query_max_length,
            document_max_length=config.document_max_length,
        )
        state = torch.load(checkpoint_path / "retrieval_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def save(self, output_dir: str, tokenizer=None) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "retrieval_model.pt")
        with (path / "retrieval_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.retrieval_config), handle, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    def _hidden_states(self, batch: dict) -> torch.Tensor:
        inputs = {key: value for key, value in batch.items() if key in {"input_ids", "attention_mask"}}
        if "attention_mask" in inputs:
            global_attention_mask = torch.zeros_like(inputs["attention_mask"])
            global_attention_mask[:, 0] = 1
            inputs["global_attention_mask"] = global_attention_mask
        return self.encoder(**inputs, return_dict=True).last_hidden_state

    def encode_single(self, batch: dict) -> torch.Tensor:
        hidden = self.projection(self._hidden_states(batch))
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if self.retrieval_config.pooling == "cls":
            embeddings = hidden[:, 0]
        elif self.retrieval_config.pooling == "mean":
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError(f"Unknown pooling mode: {self.retrieval_config.pooling}")
        return F.normalize(embeddings, dim=-1) if self.retrieval_config.normalize else embeddings

    def encode_tokens(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.projection(self._hidden_states(batch))
        if self.retrieval_config.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        mask = batch["attention_mask"].bool()
        return embeddings, mask


def load_retrieval_tokenizer(checkpoint_or_model: str):
    return AutoTokenizer.from_pretrained(checkpoint_or_model, use_fast=True)


def tokenize_texts(tokenizer, texts: Sequence[str], max_length: int, device: torch.device) -> dict:
    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def encode_single_texts(
    model: LongContextRetriever,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    return model.encode_single(tokenize_texts(tokenizer, texts, max_length, device))


@torch.inference_mode()
def encode_token_texts(
    model: LongContextRetriever,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device: torch.device,
) -> List[torch.Tensor]:
    embeddings, mask = model.encode_tokens(tokenize_texts(tokenizer, texts, max_length, device))
    return [embedding[current_mask].cpu() for embedding, current_mask in zip(embeddings, mask)]


def maxsim_scores(
    query_embeddings: torch.Tensor,
    query_mask: torch.Tensor,
    document_embeddings: torch.Tensor,
    document_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute ColBERT MaxSim scores for all query/document pairs in a batch."""
    similarities = torch.einsum("bqd,ckd->bcqk", query_embeddings, document_embeddings)
    similarities = similarities.masked_fill(~document_mask[None, :, None, :], torch.finfo(similarities.dtype).min)
    token_scores = similarities.max(dim=-1).values
    token_scores = token_scores.masked_fill(~query_mask[:, None, :], 0.0)
    return token_scores.sum(dim=-1)
