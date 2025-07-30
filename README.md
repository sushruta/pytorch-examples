# pytorch-examples
learn and do stuff in low level pytorch

# List of Examples

- [Large Model From Scratch](#large-model-from-scratch)

## Large Model From Scratch

This example demonstrates how to train a large language model from scratch using PyTorch. It showcases the use of Fully Sharded Data Parallel (FSDP) for efficient training on multiple GPUs and nodes. The code is designed to be modular and configurable, allowing you to experiment with different model sizes and training setups.

### Project Structure

- `model.py`: Defines the `SimpleGPT` model, a GPT-2 like architecture.
- `train.py`: The main training script that handles FSDP setup, training loop, and profiling.
- `dataloader.py` & `dataset.py`: Utilities for creating a synthetic dataset and dataloader for training.
- `workload.py`: Contains configurations for various model sizes (e.g., 1B, 8B, 70B parameters).
- `run-single-gpu.sh`, `run-multigpu.sh`, `run-distributed.sh`: Scripts to run the training in different environments.

### How to Run

First, navigate to the project directory and install the dependencies:
```bash
cd large-model-from-scratch
uv venv .venv
uv pip install -e .
```

#### Single GPU

To run the training on a single GPU, use the `run-single-gpu.sh` script. You need to set the `TORCHRUN_MODEL_SIZE` and `TORCHRUN_PER_DEVICE_BATCH_SIZE` environment variables.

```bash
export TORCHRUN_MODEL_SIZE="1B"
export TORCHRUN_PER_DEVICE_BATCH_SIZE=8
./run-single-gpu.sh
```

#### Multi-GPU (Single Node)

For multi-GPU training on a single node, use the `run-multigpu.sh` script. This will use all available GPUs on the node.

```bash
export TORCHRUN_MODEL_SIZE="8B"
export TORCHRUN_PER_DEVICE_BATCH_SIZE=4
./run-multigpu.sh
```

#### Distributed Training (Multi-Node)

To run distributed training across multiple nodes, use the `run-distributed.sh` script. You'll need to set additional environment variables to specify the number of nodes, GPUs per node, and the master node's address and port for process group initialization.

Run the following command on all nodes:
```bash
export TORCHRUN_MODEL_SIZE="70B"
export TORCHRUN_PER_DEVICE_BATCH_SIZE=1
export TORCHRUN_GPUS_PER_NODE=8
export TORCHRUN_NUM_NODES=2
export MASTER_ADDR="<master_node_ip>"
export MASTER_PORT="29500"

./run-distributed.sh
```
```