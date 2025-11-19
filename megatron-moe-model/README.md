# Mixture of Experts (MoE) Model

## Features

- Mixed dense and MoE layers for efficient scaling
- 1.3B (single GPU) to 70B+ (16+ GPUs)
- Distribute experts across GPUs (2 experts per GPU)
- trton based attention via `torch.nn.functional.scaled_dot_product_attention`
- BF16 and FP8 support
- Optional model compilation for faster execution

## Model Configurations

| Model | Parameters | Layers | Hidden Size | Experts | GPUs | Memory |
|-------|-----------|--------|-------------|---------|------|--------|
| Small | 1.3B | 12 | 2048 | 8 | 1 | ~5GB |
| Medium | 7B | 30 | 4096 | 8 | 2-4 | ~28GB |
| Large | 16B | 40 | 5120 | 8 | 8 | ~64GB |
| XLarge | 70B+ | 60 | 8192 | 8 | 16+ | ~280GB |

## Installation

```bash
# Clone the repository
cd megatron-moe-model

# Install dependencies
uv pip install -r requirements.txt
```

## Quick Start

### Single GPU Training (1.3B model)

```bash
# Train on single GPU (no distributed communication)
bash scripts/train_single_gpu.sh
```

### Multi-GPU Training (7B model on 4 GPUs)

```bash
# Train with expert parallelism
bash scripts/train_7b_4gpu.sh
```

### Large Scale Training (16B model on 8 GPUs)

```bash
# Expert parallelism + Data parallelism
bash scripts/train_16b_8gpu.sh
```
## Architecture Details

### MoE Layer Design

Architecture features:
- **Mixed Layers**: Dense FFN layers interspersed with MoE layers
- **Top-2 Routing**: Each token is routed to 2 out of 8 experts
- **Load Balancing**: Auxiliary loss encourages uniform expert utilization
- **SwiGLU Activation**: Used in both dense and expert FFNs

### Attention Mechanism

- **Multi-Head Attention (MHA)**: Standard for smaller models
- **Grouped Query Attention (GQA)**: Optional for larger models (70B+)
- **RoPE**: Rotary Position Embeddings for better length generalization
- **Flash Attention**: Efficient implementation via SDPA

### Training Features

- **Gradient Accumulation**: Support for large effective batch sizes
- **Learning Rate Scheduling**: Warmup + Cosine decay
- **Mixed Precision**: BF16 by default, FP8 support
- **Expert Parallelism**: Distribute experts across GPUs
- **Data Parallelism**: Replicate model with different data shards

## Configuration

Models are configured via YAML files in `configs/`. Key parameters:

```yaml
model:
  hidden_size: 2048
  num_layers: 12
  num_experts: 8
  num_experts_per_tok: 2
  moe_layer_indices: [2, 4, 6, 8]  # Which layers are MoE

training:
  batch_size: 8
  sequence_length: 2048
  precision: "bf16"
  use_torch_compile: true
  expert_parallel_size: 4  # 2 experts per GPU
```

## Distributed Training

### Expert Parallelism

Experts are distributed across GPUs. With 8 experts and `expert_parallel_size=4`:
- GPU 0: Experts 0-1
- GPU 1: Experts 2-3
- GPU 2: Experts 4-5
- GPU 3: Experts 6-7

### Data Parallelism

Multiple replicas process different data batches. Gradients are averaged across replicas.

### Example: 8 GPU Setup

```yaml
training:
  expert_parallel_size: 4  # 4 GPUs for experts
  data_parallel_size: 2     # 2 data parallel replicas
```

This creates 2 groups:
- Group 0: GPUs 0-3 (same data, different experts)
- Group 1: GPUs 4-7 (same data, different experts)

## Project Structure

```
megatron-moe-model/
├── configs/              # Model configurations
│   ├── model_1_3b.yaml
│   ├── model_7b.yaml
│   ├── model_16b.yaml
│   └── model_70b.yaml
├── models/               # Model implementation
│   ├── attention.py      # Attention layers with RoPE
│   ├── router.py         # Top-K routing
│   ├── moe_layer.py      # MoE layer implementation
│   └── moe_model.py      # Full model
├── training/             # Training utilities
│   └── distributed.py    # Distributed setup
├── scripts/              # Launch scripts
├── train.py              # Training script
├── inference.py          # Inference script
└── README.md
```

### Multi-Node Training

For 70B model on 2 nodes (16 GPUs total):

```bash
# Node 0
NODE_RANK=0 MASTER_ADDR=node0_ip bash scripts/train_70b_16gpu.sh

# Node 1
NODE_RANK=1 MASTER_ADDR=node0_ip bash scripts/train_70b_16gpu.sh
```

