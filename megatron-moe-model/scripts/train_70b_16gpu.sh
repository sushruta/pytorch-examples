#!/bin/bash

# Training script for 70B model on 16 GPUs
# Expert parallel size: 4, Data parallel size: 4

MODEL_CONFIG="configs/model_70b.yaml"
OUTPUT_DIR="outputs/70b_16gpu"
NUM_GPUS=16

echo "Training 70B+ MoE model on 16 GPUs..."
echo "Model config: $MODEL_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Expert parallel size: 4, Data parallel size: 4"

# For multi-node training, adjust --nnodes, --node_rank
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    train.py \
    --config $MODEL_CONFIG \
    --output_dir $OUTPUT_DIR \
    --seed 42

echo "Training completed!"
