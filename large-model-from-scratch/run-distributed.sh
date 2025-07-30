set -eux

uv venv .venv
uv pip install -e .

if [[ ! -v TORCHRUN_GPUS_PER_NODE ]]; then
    echo "TORCHRUN_GPUS_PER_NODE is not set"
    exit 1
fi

if [[ ! -v TORCHRUN_NUM_NODES ]]; then
    echo "TORCHRUN_NUM_NODES is not set"
    exit 1
fi

if [[ ! -v TORCHRUN_MODEL_SIZE ]]; then
    echo "TORCHRUN_MODEL_SIZE is not set"
    exit 1
fi

if [[ ! -v TORCHRUN_PER_DEVICE_BATCH_SIZE ]]; then
    echo "TORCHRUN_PER_DEVICE_BATCH_SIZE is not set"
    exit 1
fi

uv run torchrun \
  --nproc_per_node=${TORCHRUN_GPUS_PER_NODE} \
  --nnodes=${TORCHRUN_NUM_NODES} \
  train.py \
  --model_size=${TORCHRUN_MODEL_SIZE} \
  --per_device_batch_size=${TORCHRUN_PER_DEVICE_BATCH_SIZE}
