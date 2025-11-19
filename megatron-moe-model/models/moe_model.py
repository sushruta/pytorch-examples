"""
MoE Transformer Model.
Complete model implementation with mixed dense and MoE layers.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .attention import Attention, RMSNorm
from .moe_layer import MLP, MoELayer


@dataclass
class ModelConfig:
    """Configuration for MoE model."""

    # Model name
    name: str = None

    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    intermediate_size: int = 5504
    max_position_embeddings: int = 4096

    # MoE configuration
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_indices: List[int] = None
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    # Architecture details
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_flash_attention: bool = True

    # Expert parallelism
    expert_parallel_group: Optional[object] = None

    def __post_init__(self):
        if self.moe_layer_indices is None:
            # Default: every 3rd layer is MoE
            self.moe_layer_indices = list(range(2, self.num_layers, 3))

        # Generate default name if not provided
        if self.name is None:
            self.name = (
                f"moe-{self.num_layers}L-{self.hidden_size}H-"
                f"{self.num_experts}E-{self.num_experts_per_tok}K"
            )


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and FFN.
    Can be either dense (standard MLP) or sparse (MoE).
    """

    def __init__(self, config: ModelConfig, is_moe_layer: bool = False):
        super().__init__()
        self.config = config
        self.is_moe_layer = is_moe_layer

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Self-attention
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            use_flash_attention=config.use_flash_attention,
        )

        # Pre-FFN norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN (either dense MLP or MoE)
        if is_moe_layer:
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                router_aux_loss_coef=config.router_aux_loss_coef,
                router_z_loss_coef=config.router_z_loss_coef,
                hidden_dropout=config.hidden_dropout,
                expert_parallel_group=config.expert_parallel_group,
            )
        else:
            self.mlp = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_dropout=config.hidden_dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            output: [batch_size, seq_len, hidden_size]
            router_loss: Optional router loss (only for MoE layers)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            hidden_states, router_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            router_loss = None

        hidden_states = residual + hidden_states

        return hidden_states, router_loss


class MyModel(nn.Module):
    """
    MoE Transformer model.
    Mixes dense and sparse (MoE) layers for efficient scaling.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers (mix of dense and MoE)
        self.layers = nn.ModuleList([
            TransformerBlock(
                config,
                is_moe_layer=(i in config.moe_layer_indices)
            )
            for i in range(config.num_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between embedding and output
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for language modeling loss

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: Optional language modeling loss
            aux_loss: Optional auxiliary loss from MoE routing
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Accumulate router losses from MoE layers
        total_aux_loss = 0.0
        num_moe_layers = 0

        # Forward through transformer layers
        for layer in self.layers:
            hidden_states, router_loss = layer(hidden_states, attention_mask)

            if router_loss is not None:
                total_aux_loss += router_loss
                num_moe_layers += 1

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)

            loss = loss_fct(shift_logits, shift_labels)

        # Average auxiliary loss across MoE layers
        aux_loss = total_aux_loss / num_moe_layers if num_moe_layers > 0 else None

        return logits, loss, aux_loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get logits for next token
            with torch.no_grad():
                logits, _, _ = self.forward(input_ids)
                logits = logits[:, -1, :]  # Take last token logits

                # Apply temperature
                logits = logits / temperature

                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')

                # Top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float('inf')

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU).

        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds

        Returns:
            MFU as a fraction of peak FLOPS
        """
        # Rough estimate of FLOPS per token
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_layers, cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.max_position_embeddings

        # Attention: 4 * H * Q * T^2
        # FFN: 8 * hidden_size * intermediate_size
        # Rough estimate: 6N per token (forward + backward ≈ 3x forward)
        flops_per_token = 6 * N
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter

        # Express as ratio of H100 peak FLOPS (989 TFLOPS for BF16)
        flops_achieved = flops_per_iter / dt
        flops_promised = 989e12  # H100 SXM peak BF16
        mfu = flops_achieved / flops_promised

        return mfu


def load_config_from_yaml(config_path: str) -> ModelConfig:
    """Load model configuration from YAML file."""
    import yaml

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    model_config = config_dict.get('model', {})

    return ModelConfig(**model_config)
