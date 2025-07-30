import dataclasses

@dataclasses.dataclass
class BaseWorkloadConfig:
    # shared defaults
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 2048
    n_head: int = 16
    n_layer: int = 20
    gpt2_layer_norm_epsilon: float = 1e-5
    gpt2_attn_pdrop: float = 0.1
    gpt2_resid_pdrop: float = 0.1
    gpt2_embd_pdrop: float = 0.1
    num_samples: int = 1024
    seq_len: int = 512
    gradient_accumulation_steps: int = 8
    max_epochs: int = 4800
    per_device_batch_size: int = 8
    dataloader_num_workers: int = 16
    learning_rate: float = 2 * 1e-4
    log_interval: int = 50


@dataclasses.dataclass
class WorkloadConfig1B(BaseWorkloadConfig):
    pass


@dataclasses.dataclass
class WorkloadConfig2B(BaseWorkloadConfig):
    n_layer: int = 40


@dataclasses.dataclass
class WorkloadConfig4B(BaseWorkloadConfig):
    n_embd: int = 2560
    n_head: int = 20
    n_layer: int = 50


@dataclasses.dataclass
class WorkloadConfig8B(BaseWorkloadConfig):
    n_embd: int = 4096
    n_head: int = 32
    n_layer: int = 40


@dataclasses.dataclass
class WorkloadConfig16B(BaseWorkloadConfig):
    n_embd: int = 5120
    n_head: int = 40
    n_layer: int = 50


@dataclasses.dataclass
class WorkloadConfig32B(BaseWorkloadConfig):
    n_embd: int = 6144
    n_head: int = 48
    n_layer: int = 70


@dataclasses.dataclass
class WorkloadConfig70B(BaseWorkloadConfig):
    n_embd: int = 8192
    n_head: int = 64
    n_layer: int = 80

