#!/bin/bash

# Training script for single GPU (1.3B model)
# No distributed setup needed

MODEL_CONFIG="configs/model_1_3b.yaml"
OUTPUT_DIR="outputs/1_3b_single_gpu"

echo "Training 1.3B MoE model on single GPU..."
echo "Model config: $MODEL_CONFIG"
echo "Output directory: $OUTPUT_DIR"

python train.py \
    --config $MODEL_CONFIG \
    --output_dir $OUTPUT_DIR \
    --seed 42

echo "Training completed!"
