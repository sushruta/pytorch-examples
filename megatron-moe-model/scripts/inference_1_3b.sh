#!/bin/bash

# Inference script for 1.3B model

MODEL_CONFIG="configs/model_1_3b.yaml"
CHECKPOINT=""  # Add path to checkpoint if available
PROMPT="Once upon a time in a land far away"

echo "Running inference with 1.3B MoE model..."
echo "Prompt: $PROMPT"

python inference.py \
    --config $MODEL_CONFIG \
    --mode generate \
    --prompt "$PROMPT" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9 \
    --device cuda

echo "Inference completed!"
