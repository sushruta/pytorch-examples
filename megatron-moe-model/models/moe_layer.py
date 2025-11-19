"""
MoE Layer implementation with expert parallelism support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .router import TopKRouter


class MLP(nn.Module):
    """
    Standard MLP/FFN block used in both dense layers and as experts in MoE layers.
    Uses SwiGLU activation.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU requires two projections for gating
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: Swish(x @ W_gate) * (x @ W_up)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        output = self.dropout(output)
        return output


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with Top-K routing.
    Supports expert parallelism via torch.distributed.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        hidden_dropout: float = 0.0,
        expert_parallel_group: Optional[object] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_parallel_group = expert_parallel_group

        # Router
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            aux_loss_coef=router_aux_loss_coef,
            z_loss_coef=router_z_loss_coef,
        )

        # Determine which experts this rank owns
        if expert_parallel_group is not None:
            import torch.distributed as dist
            self.ep_size = dist.get_world_size(expert_parallel_group)
            self.ep_rank = dist.get_rank(expert_parallel_group)
            self.num_local_experts = num_experts // self.ep_size
            self.local_expert_indices = range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts
            )
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.num_local_experts = num_experts
            self.local_expert_indices = range(num_experts)

        # Create local experts
        self.experts = nn.ModuleList([
            MLP(hidden_size, intermediate_size, hidden_dropout)
            for _ in range(self.num_local_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
            router_loss: scalar tensor with auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_shape = hidden_states.shape

        # Flatten tokens for easier processing
        hidden_states = hidden_states.view(-1, hidden_size)  # [num_tokens, hidden_size]
        num_tokens = hidden_states.shape[0]

        # Route tokens to experts
        routing_weights, selected_experts, router_loss = self.router(
            hidden_states.view(batch_size, seq_len, hidden_size)
        )

        # Flatten routing information
        routing_weights = routing_weights.view(-1, self.num_experts_per_tok)  # [num_tokens, top_k]
        selected_experts = selected_experts.view(-1, self.num_experts_per_tok)  # [num_tokens, top_k]

        # Initialize output tensor
        output = torch.zeros_like(hidden_states)

        # Process tokens through selected experts
        if self.expert_parallel_group is None:
            # No expert parallelism - simple implementation
            output = self._forward_single_device(
                hidden_states, routing_weights, selected_experts
            )
        else:
            # Expert parallelism enabled
            output = self._forward_expert_parallel(
                hidden_states, routing_weights, selected_experts
            )

        # Reshape output back
        output = output.view(*original_shape)

        return output, router_loss

    def _forward_single_device(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass when all experts are on the same device."""
        num_tokens = hidden_states.shape[0]
        output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]

            # Find all tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)
            token_indices, top_k_indices = torch.where(expert_mask)

            if token_indices.numel() == 0:
                continue

            # Get tokens and their routing weights for this expert
            expert_input = hidden_states[token_indices]
            expert_weights = routing_weights[token_indices, top_k_indices].unsqueeze(-1)

            # Process through expert
            expert_output = expert(expert_input)

            # Weight and accumulate
            output[token_indices] += expert_weights * expert_output

        return output

    def _forward_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with expert parallelism using all-to-all communication."""
        import torch.distributed as dist

        num_tokens = hidden_states.shape[0]
        output = torch.zeros_like(hidden_states)

        # For simplicity, we process local experts and use all-reduce
        # A more optimized version would use all-to-all to exchange tokens

        # Process local experts
        for local_idx, expert_idx in enumerate(self.local_expert_indices):
            expert = self.experts[local_idx]

            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)
            token_indices, top_k_indices = torch.where(expert_mask)

            if token_indices.numel() == 0:
                continue

            # Process tokens
            expert_input = hidden_states[token_indices]
            expert_weights = routing_weights[token_indices, top_k_indices].unsqueeze(-1)
            expert_output = expert(expert_input)

            # Accumulate weighted output
            output[token_indices] += expert_weights * expert_output

        # All-reduce to collect outputs from all expert parallel ranks
        dist.all_reduce(output, group=self.expert_parallel_group)

        return output


class SparseMoELayer(nn.Module):
    """
    Optimized sparse MoE implementation with capacity factor and token dropping.
    Uses expert capacity to limit the number of tokens per expert.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        capacity_factor: float = 1.25,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.capacity_factor = capacity_factor

        # This would implement a more memory-efficient sparse MoE
        # For now, we'll use the standard MoELayer
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            **kwargs
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.moe(hidden_states)
