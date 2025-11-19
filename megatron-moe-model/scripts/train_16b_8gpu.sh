#!/bin/bash

# Training script for 16B model on 8 GPUs
# Expert parallel size: 4, Data parallel size: 2

MODEL_CONFIG="configs/model_16b.yaml"
OUTPUT_DIR="outputs/16b_8gpu"
NUM_GPUS=8

echo "Training 16B MoE model on 8 GPUs..."
echo "Model config: $MODEL_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Expert parallel size: 4, Data parallel size: 2"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py \
    --config $MODEL_CONFIG \
    --output_dir $OUTPUT_DIR \
    --seed 42

echo "Training completed!"
