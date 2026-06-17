from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention

from graphbert.config import GraphAttentionConfig


class GraphBertSelfAttention(BertSelfAttention):
    """BERT self-attention followed by per-head GCN propagation.

    The vanilla BERT attention graph is reused as a dynamic adjacency matrix.
    After standard attention computes per-head context vectors, the sparsified
    attention matrix propagates those vectors once more via H' = A H W.
    """

    def __init__(self, bert_config, graph_config: GraphAttentionConfig, position_embedding_type=None):
        try:
            super().__init__(bert_config, position_embedding_type=position_embedding_type)
        except TypeError:
            super().__init__(bert_config)
            if position_embedding_type is not None and hasattr(self, "position_embedding_type"):
                self.position_embedding_type = position_embedding_type
        self.graph_config = graph_config
        self.gcn_weight = nn.Parameter(
            torch.empty(self.num_attention_heads, self.attention_head_size, self.attention_head_size)
        )
        self.latest_graph_stats = {}
        self.reset_graph_parameters()

    def reset_graph_parameters(self) -> None:
        if self.graph_config.gcn_weight_init == "identity":
            with torch.no_grad():
                self.gcn_weight.zero_()
                eye = torch.eye(self.attention_head_size)
                self.gcn_weight.copy_(eye.unsqueeze(0).repeat(self.num_attention_heads, 1, 1))
        else:
            nn.init.xavier_uniform_(self.gcn_weight)

    def transpose_for_scores(self, tensor: torch.Tensor) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    @classmethod
    def from_bert_self_attention(
        cls, source: BertSelfAttention, bert_config, graph_config: GraphAttentionConfig
    ) -> "GraphBertSelfAttention":
        module = cls(
            bert_config,
            graph_config=graph_config,
            position_embedding_type=getattr(source, "position_embedding_type", None),
        )
        module.load_state_dict(source.state_dict(), strict=False)
        return module

    def _dense_propagate(self, weights: torch.Tensor, context_heads: torch.Tensor) -> torch.Tensor:
        if self.graph_config.add_self_loops:
            eye = torch.eye(weights.size(-1), device=weights.device, dtype=weights.dtype)
            weights = weights + eye.view(1, 1, weights.size(-1), weights.size(-1))

        if self.graph_config.symmetric_normalization:
            degree = weights.sum(dim=-1).clamp_min(1e-12)
            weights = weights * degree.rsqrt().unsqueeze(-1) * degree.rsqrt().unsqueeze(-2)
        elif self.graph_config.renormalize_adjacency:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        self._record_dense_graph_stats(weights.detach())
        return torch.matmul(weights, context_heads)

    def _topk_propagate(self, attention_probs: torch.Tensor, context_heads: torch.Tensor) -> torch.Tensor:
        k = min(self.graph_config.top_k, attention_probs.size(-1))
        values, indices = torch.topk(attention_probs, k=k, dim=-1)
        original_values = values

        batch_size, num_heads, query_length, _, head_dim = (
            context_heads.size(0),
            context_heads.size(1),
            context_heads.size(2),
            values.size(-1),
            context_heads.size(-1),
        )

        expanded_context = context_heads.unsqueeze(2).expand(
            batch_size, num_heads, query_length, context_heads.size(2), head_dim
        )
        neighbor_states = torch.gather(
            expanded_context,
            dim=3,
            index=indices.unsqueeze(-1).expand(batch_size, num_heads, query_length, k, head_dim),
        )

        if self.graph_config.add_self_loops:
            self_indices = torch.arange(query_length, device=indices.device).view(1, 1, query_length, 1)
            self_indices = self_indices.expand(batch_size, num_heads, query_length, 1)
            self_states = context_heads.unsqueeze(3)
            indices = torch.cat([indices, self_indices], dim=-1)
            values = torch.cat([values, torch.ones_like(values[..., :1])], dim=-1)
            neighbor_states = torch.cat([neighbor_states, self_states], dim=-2)

        if self.graph_config.symmetric_normalization:
            degree = torch.zeros(
                batch_size,
                num_heads,
                query_length,
                device=attention_probs.device,
                dtype=attention_probs.dtype,
            )
            degree.scatter_add_(-1, indices.reshape(batch_size, num_heads, -1), values.reshape(batch_size, num_heads, -1))
            row_degree = values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            col_degree = torch.gather(
                degree.clamp_min(1e-12).unsqueeze(2).expand(batch_size, num_heads, query_length, query_length),
                dim=-1,
                index=indices,
            )
            values = values * row_degree.rsqrt() * col_degree.rsqrt()
        elif self.graph_config.renormalize_adjacency:
            values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        self._record_topk_graph_stats(attention_probs.detach(), original_values.detach())
        return (neighbor_states * values.unsqueeze(-1)).sum(dim=-2)

    def _propagate_graph(self, attention_probs: torch.Tensor, context_heads: torch.Tensor) -> torch.Tensor:
        strategy = self.graph_config.sparsification
        if strategy == "dense":
            return self._dense_propagate(attention_probs, context_heads)
        if strategy == "threshold":
            return self._dense_propagate(
                attention_probs.masked_fill(attention_probs < self.graph_config.threshold, 0.0),
                context_heads,
            )
        if strategy == "topk":
            return self._topk_propagate(attention_probs, context_heads)
        raise ValueError(f"Unknown sparsification strategy: {strategy}")

    @torch.no_grad()
    def _record_dense_graph_stats(self, weights: torch.Tensor) -> None:
        surviving = weights > 0
        num_edges = surviving.sum().item()
        total_edges = surviving.numel()
        degree = surviving.sum(dim=-1).float()
        self.latest_graph_stats = {
            "graph_sparsity": 1.0 - (num_edges / max(total_edges, 1)),
            "graph_avg_degree": degree.mean().item(),
            "graph_surviving_edge_pct": 100.0 * num_edges / max(total_edges, 1),
        }

    @torch.no_grad()
    def _record_topk_graph_stats(self, raw_attention: torch.Tensor, values: torch.Tensor) -> None:
        surviving = values > 0
        num_edges = surviving.sum().item()
        total_edges = raw_attention.numel()
        possible_edges = (raw_attention > 0).sum().clamp_min(1).item()
        degree = surviving.sum(dim=-1).float()
        self.latest_graph_stats = {
            "graph_sparsity": 1.0 - (num_edges / max(total_edges, 1)),
            "graph_avg_degree": degree.mean().item(),
            "graph_surviving_edge_pct": 100.0 * num_edges / max(possible_edges, 1),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        if isinstance(past_key_values, bool):
            output_attentions = past_key_values
            past_key_values = None
        if isinstance(past_key_value, bool):
            output_attentions = past_key_value
            past_key_value = None
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values

        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        position_embedding_type = getattr(self, "position_embedding_type", "absolute")
        if position_embedding_type in {"relative_key", "relative_key_query"}:
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_heads = torch.matmul(attention_probs, value_layer)
        propagated = self._propagate_graph(attention_probs, context_heads)
        propagated = torch.einsum("bhld,hdf->bhlf", propagated, self.gcn_weight)

        context_layer = propagated.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # Transformers 5 BertAttention unpacks `(attention_output, attn_weights)`;
        # Transformers 4 treats extra tuple items as optional attention outputs.
        outputs = (context_layer, attention_probs)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
