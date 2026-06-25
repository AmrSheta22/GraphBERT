from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.longformer.modeling_longformer import LongformerLayer

from graphbert.config import GraphAttentionConfig


def _sliding_sum(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Sum values in an inclusive sliding window without materializing its edges."""
    original_dtype = values.dtype
    if values.dtype in {torch.float16, torch.bfloat16}:
        values = values.float()
    padded = nn.functional.pad(values, (0, 0, radius, radius))
    prefix = nn.functional.pad(padded.cumsum(dim=1), (0, 0, 1, 0))
    width = 2 * radius + 1
    return (prefix[:, width:] - prefix[:, :-width]).to(dtype=original_dtype)


class APPNPLongformerLayer(nn.Module):
    """Intact Longformer layer with a gated sparse-APPNP residual adapter.

    The wrapped pretrained layer is always evaluated normally. Its output is
    augmented with ``alpha * APPNP(input)`` over Longformer's local/global
    topology. With alpha initialized to zero, the wrapper is exactly equivalent
    to the original layer at initialization.
    """

    def __init__(
        self,
        base_layer: LongformerLayer,
        hidden_size: int,
        attention_window: int,
        graph_config: GraphAttentionConfig,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.graph_config = graph_config
        self.attention_window = attention_window
        self.one_sided_window = attention_window // 2
        self.appnp_projection = nn.Linear(hidden_size, hidden_size, bias=graph_config.appnp_bias)
        self.appnp_dropout = nn.Dropout(graph_config.appnp_dropout)
        self.activation = (
            ACT2FN[graph_config.appnp_activation]
            if graph_config.appnp_activation != "none"
            else None
        )
        gate_shape = (hidden_size,) if graph_config.appnp_gate_type == "channel" else ()
        self.appnp_gate = nn.Parameter(torch.full(gate_shape, graph_config.appnp_initial_scale))
        self.latest_graph_stats = {}
        self.reset_appnp_parameters()

    def reset_appnp_parameters(self) -> None:
        if self.graph_config.appnp_weight_init == "identity":
            with torch.no_grad():
                self.appnp_projection.weight.copy_(
                    torch.eye(self.appnp_projection.out_features, dtype=self.appnp_projection.weight.dtype)
                )
                if self.appnp_projection.bias is not None:
                    self.appnp_projection.bias.zero_()
        else:
            nn.init.xavier_uniform_(self.appnp_projection.weight)
            if self.appnp_projection.bias is not None:
                nn.init.zeros_(self.appnp_projection.bias)

    @classmethod
    def from_longformer_layer(
        cls,
        source: LongformerLayer,
        model_config,
        layer_index: int,
        graph_config: GraphAttentionConfig,
    ) -> "APPNPLongformerLayer":
        windows = model_config.attention_window
        attention_window = windows[layer_index] if isinstance(windows, (list, tuple)) else windows
        return cls(source, model_config.hidden_size, attention_window, graph_config)

    def _degrees(self, valid: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
        local_source = (valid & ~global_mask).to(dtype=torch.float32).unsqueeze(-1)
        local_degree = _sliding_sum(local_source, self.one_sided_window).squeeze(-1)
        global_count = (valid & global_mask).sum(dim=1, keepdim=True).to(dtype=torch.float32)
        valid_count = valid.sum(dim=1, keepdim=True).to(dtype=torch.float32)
        degree = torch.where(global_mask, valid_count, local_degree + global_count)

        if not self.graph_config.add_self_loops:
            degree = degree - valid.to(dtype=degree.dtype)
        return degree.clamp_min(0.0) * valid.to(dtype=degree.dtype)

    def _propagate_once(
        self,
        states: torch.Tensor,
        valid: torch.Tensor,
        global_mask: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = states.dtype
        source_states = states.float() if states.dtype in {torch.float16, torch.bfloat16} else states
        if self.graph_config.symmetric_normalization:
            source_states = source_states * degree.clamp_min(1.0).rsqrt().unsqueeze(-1)

        local_source_mask = (valid & ~global_mask).unsqueeze(-1)
        local_sum = _sliding_sum(source_states * local_source_mask, self.one_sided_window)
        global_sum = (source_states * (valid & global_mask).unsqueeze(-1)).sum(dim=1, keepdim=True)
        all_valid_sum = (source_states * valid.unsqueeze(-1)).sum(dim=1, keepdim=True)

        aggregated = torch.where(global_mask.unsqueeze(-1), all_valid_sum, local_sum + global_sum)
        if not self.graph_config.add_self_loops:
            aggregated = aggregated - source_states * valid.unsqueeze(-1)

        if self.graph_config.symmetric_normalization:
            aggregated = aggregated * degree.clamp_min(1.0).rsqrt().unsqueeze(-1)
        elif self.graph_config.renormalize_adjacency:
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)

        aggregated = aggregated * valid.unsqueeze(-1)
        return aggregated.to(dtype=original_dtype)

    def _appnp(
        self,
        hidden_states: torch.Tensor,
        valid: torch.Tensor,
        global_mask: torch.Tensor,
    ) -> torch.Tensor:
        degree = self._degrees(valid, global_mask).to(device=hidden_states.device)
        initial = self.appnp_projection(hidden_states)
        if self.activation is not None:
            initial = self.activation(initial)
        initial = self.appnp_dropout(initial) * valid.unsqueeze(-1)

        propagated = initial
        teleport = self.graph_config.appnp_teleport_probability
        for _ in range(self.graph_config.appnp_steps):
            propagated = (
                (1.0 - teleport) * self._propagate_once(propagated, valid, global_mask, degree)
                + teleport * initial
            )
        self._record_graph_stats(degree, valid)
        return propagated * valid.unsqueeze(-1)

    @torch.no_grad()
    def _record_graph_stats(self, degree: torch.Tensor, valid: torch.Tensor) -> None:
        valid_nodes = valid.sum().detach()
        edges = degree.sum().detach()
        self.latest_graph_stats = {
            "graph_avg_degree": edges / valid_nodes.clamp_min(1),
            "graph_edges": edges,
            "graph_valid_nodes": valid_nodes,
            "graph_residual_scale": self.appnp_gate.detach().abs().mean(),
            "appnp_steps": torch.tensor(
                float(self.graph_config.appnp_steps),
                device=degree.device,
            ),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_index_masked: Optional[torch.Tensor] = None,
        is_index_global_attn: Optional[torch.Tensor] = None,
        is_global_attn: Optional[bool] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        base_outputs = self.base_layer(
            hidden_states,
            attention_mask=attention_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
            **kwargs,
        )

        batch_size, sequence_length = hidden_states.shape[:2]
        if is_index_masked is None:
            valid = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=hidden_states.device)
        else:
            valid = ~is_index_masked
        if is_index_global_attn is None:
            global_mask = torch.zeros_like(valid)
        else:
            global_mask = is_index_global_attn & valid

        appnp_output = self._appnp(hidden_states, valid, global_mask)
        adapted_output = base_outputs[0] + self.appnp_gate * appnp_output
        return (adapted_output,) + base_outputs[1:]


# Compatibility alias for older imports.
GraphLongformerLayer = APPNPLongformerLayer
