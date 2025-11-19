#!/bin/bash

# Training script for 7B model on 4 GPUs with expert parallelism
# Expert parallel size: 4 (2 experts per GPU)

MODEL_CONFIG="configs/model_7b.yaml"
OUTPUT_DIR="outputs/7b_4gpu"
NUM_GPUS=4

echo "Training 7B MoE model on 4 GPUs with expert parallelism..."
echo "Model config: $MODEL_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"

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
