import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from workload import BaseWorkloadConfig


class SimpleGPT(nn.Module):
    def __init__(self, wconf: BaseWorkloadConfig):
        super().__init__()
        gpt2_config = GPT2Config(
            vocab_size=wconf.vocab_size,
            n_positions=wconf.n_positions,
            n_embd=wconf.n_embd,
            n_head=wconf.n_head,
            n_layer=wconf.n_layer,
            layer_norm_epsilon=wconf.gpt2_layer_norm_epsilon,
            attn_pdrop=wconf.gpt2_attn_pdrop,
            resid_pdrop=wconf.gpt2_resid_pdrop,
            embd_pdrop=wconf.gpt2_embd_pdrop,
            n_inner=None,
        )

        self.wte = nn.Embedding(wconf.vocab_size, wconf.n_embd)
        self.wpe = nn.Embedding(wconf.n_positions, wconf.n_embd)
        self.blocks = nn.ModuleList(
            checkpoint_wrapper(GPT2Block(gpt2_config)) for _ in range(wconf.n_layer)
        )
        self.layernorm = nn.LayerNorm(wconf.n_embd, eps=1e-5)
        self.linear = nn.Linear(wconf.n_embd, wconf.vocab_size)

        self.wte.weight = self.linear.weight
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.LongTensor):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)

        token_embeddings = self.wte(x)
        position_embeddings = self.wpe(positions)

        hidden_states = token_embeddings + position_embeddings

        for block in self.blocks:
            hidden_states = block(hidden_states)[0]

        hidden_states = self.layernorm(hidden_states)
        logits = self.linear(hidden_states)
        return logits
