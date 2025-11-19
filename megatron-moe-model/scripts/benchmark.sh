#!/bin/bash

# Benchmark script to measure throughput

MODEL_CONFIG="configs/model_1_3b.yaml"
BATCH_SIZE=4
SEQ_LEN=512

echo "Benchmarking model throughput..."
echo "Model config: $MODEL_CONFIG"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQ_LEN"

python inference.py \
    --config $MODEL_CONFIG \
    --mode benchmark \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --device cuda

echo "Benchmark completed!"
