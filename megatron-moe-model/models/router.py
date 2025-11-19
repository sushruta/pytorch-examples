"""
MoE Router with Top-K gating and load balancing.
Implements routing with auxiliary losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TopKRouter(nn.Module):
    """
    Top-K gating router for MoE.
    Routes each token to top-K experts based on learned gating weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef

        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            routing_weights: [batch_size, seq_len, num_experts_per_tok]
            selected_experts: [batch_size, seq_len, num_experts_per_tok]
            router_loss: scalar tensor with auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Compute gating logits: [batch_size, seq_len, num_experts]
        router_logits = self.gate(hidden_states)

        # Compute routing weights and select top-k experts
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )

        # Normalize the routing weights for selected experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Compute auxiliary losses
        router_loss = self._compute_auxiliary_loss(router_logits, routing_weights, selected_experts)

        return routing_weights, selected_experts, router_loss

    def _compute_auxiliary_loss(
        self,
        router_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary losses for load balancing.

        Implements:
        1. Load balancing loss (auxiliary loss) - encourages uniform distribution across experts
        2. Router z-loss - encourages smaller logits for numerical stability
        """
        # Load balancing loss
        if self.aux_loss_coef > 0:
            # Compute the fraction of tokens routed to each expert
            # Create one-hot encoding of selected experts
            num_tokens = router_logits.shape[0] * router_logits.shape[1]
            expert_mask = torch.zeros_like(router_logits)
            expert_mask.scatter_(-1, selected_experts, 1)

            # Fraction of tokens routed to each expert
            tokens_per_expert = expert_mask.sum(dim=(0, 1)) / num_tokens

            # Average routing probability for each expert (use full softmax, not just top-k)
            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            router_prob_per_expert = router_probs.sum(dim=(0, 1)) / num_tokens

            # Load balancing loss: encourage equal distribution
            aux_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
        else:
            aux_loss = 0.0

        # Router z-loss: encourages smaller logits
        if self.z_loss_coef > 0:
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            z_loss = self.z_loss_coef * z_loss
        else:
            z_loss = 0.0

        return self.aux_loss_coef * aux_loss + z_loss


class ExpertChoiceRouter(nn.Module):
    """
    Alternative router where experts choose tokens instead of tokens choosing experts.
    This can provide better load balancing in some cases.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_capacity: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing.
        Each expert selects its top-k tokens.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len

        # Reshape for processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Compute routing scores
        router_logits = self.gate(hidden_states_flat)  # [num_tokens, num_experts]
        router_weights = F.softmax(router_logits, dim=0, dtype=torch.float32)

        # Each expert selects top-k tokens
        expert_weights, expert_indices = torch.topk(
            router_weights, self.expert_capacity, dim=0
        )

        return expert_weights.to(hidden_states.dtype), expert_indices
