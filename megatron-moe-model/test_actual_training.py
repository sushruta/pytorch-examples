"""
Quick test of actual training with all metrics displayed.
"""

import sys
import tempfile
import yaml

# Create a minimal config for quick testing
config = {
    'model': {
        'vocab_size': 5000,
        'hidden_size': 256,
        'num_layers': 4,
        'num_attention_heads': 4,
        'num_key_value_heads': 4,
        'intermediate_size': 688,
        'max_position_embeddings': 512,
        'num_experts': 4,
        'num_experts_per_tok': 2,
        'moe_layer_indices': [1, 3],
        'router_aux_loss_coef': 0.01,
        'router_z_loss_coef': 0.001,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
        'use_flash_attention': True,
    },
    'training': {
        'batch_size': 2,
        'sequence_length': 128,
        'gradient_accumulation_steps': 1,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'warmup_steps': 5,
        'max_steps': 25,  # Only 25 steps for demo
        'precision': 'bf16',
        'use_torch_compile': False,
        'distributed': False,
        'expert_parallel_size': 1,
        'data_parallel_size': 1,
        'log_interval': 10,  # Log every 10 steps
        'eval_interval': 1000,
    }
}

# Write config to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(config, f)
    config_path = f.name

print(f"Created test config at: {config_path}")
print("\nStarting training with full metrics...\n")
print("="*80)

# Run training
import subprocess
result = subprocess.run(
    [sys.executable, 'train.py', '--config', config_path, '--output_dir', 'test_output', '--seed', '42'],
    capture_output=False,
    text=True
)

print("="*80)
print("\nTraining completed!")
print("\nNote: All the metrics (architecture, FLOPS, throughput, etc.) are displayed above.")
