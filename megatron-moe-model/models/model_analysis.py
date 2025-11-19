"""
Model analysis utilities for printing architecture details, FLOPS, and metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .moe_model import MyModel, ModelConfig


# GPU specifications (BF16 and FP8 peak TFLOPS)
GPU_SPECS = {
    'H100': {
        'name': 'NVIDIA H100 80GB HBM3',
        'bf16_tflops': 989,
        'fp8_tflops': 1979,
        'memory_gb': 80,
    },
    'H200': {
        'name': 'NVIDIA H200 141GB HBM3e',
        'bf16_tflops': 989,
        'fp8_tflops': 1979,
        'memory_gb': 141,
    },
    'GB200': {
        'name': 'NVIDIA GB200 Grace Blackwell Superchip',
        'bf16_tflops': 1250,
        'fp8_tflops': 2500,
        'memory_gb': 192,
    },
    'GB300': {
        'name': 'NVIDIA GB300 Grace Blackwell Superchip',
        'bf16_tflops': 1500,
        'fp8_tflops': 3000,
        'memory_gb': 288,
    },
}


def detect_gpu_type() -> Optional[str]:
    """
    Detect which GPU type is being used.
    Returns GPU key from GPU_SPECS or None if unknown.
    """
    if not torch.cuda.is_available():
        return None

    gpu_name = torch.cuda.get_device_name(0).upper()

    # Check for specific GPU types
    if 'H200' in gpu_name:
        return 'H200'
    elif 'H100' in gpu_name:
        return 'H100'
    elif 'GB300' in gpu_name or 'B300' in gpu_name:
        return 'GB300'
    elif 'GB200' in gpu_name or 'B200' in gpu_name or 'BLACKWELL' in gpu_name:
        return 'GB200'

    # Default to H100 if NVIDIA GPU but not recognized
    if 'NVIDIA' in gpu_name:
        return 'H100'

    return None


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }


def analyze_layer_shapes(model: MyModel) -> Dict[str, any]:
    """Analyze shapes and sizes of model layers."""
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads

    analysis = {
        'embedding': {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'params': config.vocab_size * config.hidden_size,
        },
        'attention': {
            'num_heads': config.num_attention_heads,
            'num_kv_heads': config.num_key_value_heads,
            'head_dim': head_dim,
            'hidden_size': config.hidden_size,
            'qkv_params_per_layer': config.hidden_size * (
                config.hidden_size +  # Q projection
                2 * config.num_key_value_heads * head_dim  # K, V projections
            ),
            'output_params_per_layer': config.hidden_size * config.hidden_size,
        },
        'dense_mlp': {
            'hidden_size': config.hidden_size,
            'intermediate_size': config.intermediate_size,
            'params_per_layer': (
                config.hidden_size * config.intermediate_size * 2 +  # gate_proj + up_proj
                config.intermediate_size * config.hidden_size  # down_proj
            ),
        },
        'moe_layer': {
            'num_experts': config.num_experts,
            'num_experts_per_tok': config.num_experts_per_tok,
            'params_per_expert': (
                config.hidden_size * config.intermediate_size * 2 +
                config.intermediate_size * config.hidden_size
            ),
            'params_per_moe_layer': (
                config.num_experts * (
                    config.hidden_size * config.intermediate_size * 2 +
                    config.intermediate_size * config.hidden_size
                ) +
                config.hidden_size * config.num_experts  # router
            ),
        },
        'normalization': {
            'params_per_layer': config.hidden_size * 2,  # 2 norms per layer
        },
    }

    return analysis


def calculate_flops(config: ModelConfig, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
    """
    Calculate FLOPs for forward pass.
    Based on: https://arxiv.org/abs/2001.08361 (Kaplan et al.)
    """
    H = config.hidden_size
    L = config.num_layers
    V = config.vocab_size
    I = config.intermediate_size
    N_exp = config.num_experts
    K = config.num_experts_per_tok
    T = seq_len
    B = batch_size

    # Count MoE vs Dense layers
    num_moe_layers = len(config.moe_layer_indices)
    num_dense_layers = L - num_moe_layers

    flops = {}

    # Embedding: B * T * V * H (forward only, backward is ~2x)
    flops['embedding'] = B * T * V * H

    # Attention per layer: approximately 4 * B * T^2 * H + 8 * B * T * H^2
    # QKV projection: 3 * B * T * H^2
    # Attention weights: B * num_heads * T^2 * head_dim = B * T^2 * H
    # Attention output: B * T * H^2
    # Output projection: B * T * H^2
    attention_per_layer = (
        4 * B * T * H * H +  # QKV + output projections
        2 * B * T * T * H    # Attention computation
    )
    flops['attention_total'] = L * attention_per_layer

    # Dense FFN per layer: approximately 8 * B * T * H * I (gate + up + down)
    dense_ffn_per_layer = 8 * B * T * H * I
    flops['dense_ffn_total'] = num_dense_layers * dense_ffn_per_layer

    # MoE FFN per layer: only K out of N_exp experts are active
    # Each token goes through K experts, so: 8 * B * T * K/N_exp * N_exp * H * I = 8 * B * T * K * H * I
    moe_ffn_per_layer = 8 * B * T * K * H * I
    flops['moe_ffn_total'] = num_moe_layers * moe_ffn_per_layer

    # Router per MoE layer: B * T * H * N_exp
    router_per_layer = B * T * H * N_exp
    flops['router_total'] = num_moe_layers * router_per_layer

    # Layer norms (negligible): ~2 * B * T * H per layer
    flops['norm_total'] = 2 * L * B * T * H

    # LM head: B * T * H * V
    flops['lm_head'] = B * T * H * V

    # Total forward pass FLOPs
    flops['forward_total'] = sum(flops.values())

    # Backward pass is approximately 2x forward pass
    flops['backward_total'] = 2 * flops['forward_total']

    # Total per training step (forward + backward)
    flops['total_per_step'] = flops['forward_total'] + flops['backward_total']

    # FLOPs per token
    flops['per_token'] = flops['total_per_step'] / (B * T)

    return flops


def print_model_architecture(model: MyModel, config_dict: Dict = None):
    """Print detailed model architecture information."""
    config = model.config

    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)

    # Basic info
    model_name = getattr(config, 'name', 'MoE Model')
    print(f"\nModel: {model_name}")
    print(f"Architecture: Transformer with Mixture of Experts (MoE)")

    # Layer configuration
    print(f"\n{'LAYER CONFIGURATION':-^80}")
    print(f"  Total layers:              {config.num_layers}")
    print(f"  MoE layers:                {len(config.moe_layer_indices)} (indices: {config.moe_layer_indices[:5]}{'...' if len(config.moe_layer_indices) > 5 else ''})")
    print(f"  Dense layers:              {config.num_layers - len(config.moe_layer_indices)}")

    # Dimensions
    print(f"\n{'DIMENSIONS':-^80}")
    print(f"  Vocabulary size:           {config.vocab_size:,}")
    print(f"  Hidden size:               {config.hidden_size:,}")
    print(f"  Intermediate size (FFN):   {config.intermediate_size:,}")
    print(f"  Max sequence length:       {config.max_position_embeddings:,}")

    # Attention
    print(f"\n{'ATTENTION':-^80}")
    print(f"  Attention heads:           {config.num_attention_heads}")
    print(f"  Key/Value heads:           {config.num_key_value_heads}")
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"  Head dimension:            {head_dim}")
    if config.num_key_value_heads < config.num_attention_heads:
        print(f"  Attention type:            Grouped Query Attention (GQA)")
        print(f"  GQA groups:                {config.num_attention_heads // config.num_key_value_heads}")
    else:
        print(f"  Attention type:            Multi-Head Attention (MHA)")
    print(f"  RoPE theta:                {config.rope_theta}")
    print(f"  Flash Attention:           {'Enabled' if config.use_flash_attention else 'Disabled'}")

    # MoE Configuration
    print(f"\n{'MIXTURE OF EXPERTS':-^80}")
    print(f"  Number of experts:         {config.num_experts}")
    print(f"  Experts per token (Top-K): {config.num_experts_per_tok}")
    print(f"  Router aux loss coef:      {config.router_aux_loss_coef}")
    print(f"  Router z-loss coef:        {config.router_z_loss_coef}")

    # Parameters
    params = count_parameters(model)
    print(f"\n{'PARAMETERS':-^80}")
    print(f"  Total parameters:          {params['total']:>15,} ({params['total']/1e9:.2f}B)")
    print(f"  Trainable parameters:      {params['trainable']:>15,} ({params['trainable']/1e9:.2f}B)")
    if params['non_trainable'] > 0:
        print(f"  Non-trainable parameters:  {params['non_trainable']:>15,} ({params['non_trainable']/1e9:.2f}B)")

    # Layer-wise parameter breakdown
    shapes = analyze_layer_shapes(model)
    print(f"\n{'PARAMETER BREAKDOWN':-^80}")
    print(f"  Embedding:                 {shapes['embedding']['params']:>15,}")
    print(f"  Attention (per layer):     {shapes['attention']['qkv_params_per_layer'] + shapes['attention']['output_params_per_layer']:>15,}")
    print(f"  Dense FFN (per layer):     {shapes['dense_mlp']['params_per_layer']:>15,}")
    print(f"  MoE FFN (per layer):       {shapes['moe_layer']['params_per_moe_layer']:>15,}")
    print(f"    - Per expert:            {shapes['moe_layer']['params_per_expert']:>15,}")
    print(f"    - Router:                {config.hidden_size * config.num_experts:>15,}")

    # Memory estimate (rough)
    print(f"\n{'MEMORY ESTIMATES (BF16)':-^80}")
    # Model parameters in BF16 (2 bytes per parameter)
    model_memory_gb = params['total'] * 2 / 1e9
    # Optimizer states (AdamW: 2x parameters for momentum + variance)
    optimizer_memory_gb = params['total'] * 2 * 4 / 1e9  # FP32 optimizer states
    # Gradients (same size as parameters)
    grad_memory_gb = params['total'] * 2 / 1e9

    print(f"  Model weights:             {model_memory_gb:>10.2f} GB")
    print(f"  Gradients:                 {grad_memory_gb:>10.2f} GB")
    print(f"  Optimizer states:          {optimizer_memory_gb:>10.2f} GB")
    print(f"  Total (training):          {model_memory_gb + grad_memory_gb + optimizer_memory_gb:>10.2f} GB")
    print(f"  Inference only:            {model_memory_gb:>10.2f} GB")

    # Training configuration
    if config_dict:
        train_config = config_dict.get('training', {})
        print(f"\n{'TRAINING CONFIGURATION':-^80}")
        print(f"  Batch size:                {train_config.get('batch_size', 'N/A')}")
        print(f"  Sequence length:           {train_config.get('sequence_length', 'N/A')}")
        print(f"  Gradient accumulation:     {train_config.get('gradient_accumulation_steps', 'N/A')}")
        print(f"  Learning rate:             {train_config.get('learning_rate', 'N/A')}")
        print(f"  Precision:                 {train_config.get('precision', 'N/A')}")
        print(f"  torch.compile:             {'Enabled' if train_config.get('use_torch_compile') else 'Disabled'}")

        if train_config.get('distributed'):
            print(f"  Expert parallel size:      {train_config.get('expert_parallel_size', 1)}")
            print(f"  Data parallel size:        {train_config.get('data_parallel_size', 1)}")

    print("=" * 80 + "\n")


def print_flops_analysis(config: ModelConfig, seq_len: int, batch_size: int = 1):
    """Print FLOPS analysis."""
    flops = calculate_flops(config, seq_len, batch_size)

    print("\n" + "=" * 80)
    print("FLOPS ANALYSIS")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Batch size:                {batch_size}")
    print(f"  Sequence length:           {seq_len}")
    print(f"  Total tokens per batch:    {batch_size * seq_len:,}")

    print(f"\n{'FLOPS BREAKDOWN (Forward Pass)':-^80}")
    print(f"  Embedding:                 {flops['embedding']/1e9:>12.2f} GFLOPS")
    print(f"  Attention (all layers):    {flops['attention_total']/1e9:>12.2f} GFLOPS")
    print(f"  Dense FFN (all layers):    {flops['dense_ffn_total']/1e9:>12.2f} GFLOPS")
    print(f"  MoE FFN (all layers):      {flops['moe_ffn_total']/1e9:>12.2f} GFLOPS")
    print(f"  Router (all MoE layers):   {flops['router_total']/1e9:>12.2f} GFLOPS")
    print(f"  Layer norms:               {flops['norm_total']/1e9:>12.2f} GFLOPS")
    print(f"  LM head:                   {flops['lm_head']/1e9:>12.2f} GFLOPS")
    print(f"  {'-'*35}")
    print(f"  Total forward:             {flops['forward_total']/1e9:>12.2f} GFLOPS")

    print(f"\n{'TOTAL FLOPS (Training Step)':-^80}")
    print(f"  Forward pass:              {flops['forward_total']/1e12:>12.2f} TFLOPS")
    print(f"  Backward pass (~2x fwd):   {flops['backward_total']/1e12:>12.2f} TFLOPS")
    print(f"  Total per step:            {flops['total_per_step']/1e12:>12.2f} TFLOPS")
    print(f"  FLOPs per token:           {flops['per_token']/1e9:>12.2f} GFLOPS")

    # Hardware utilization estimates
    print(f"\n{'HARDWARE UTILIZATION ESTIMATES':-^80}")

    # Detect current GPU
    gpu_type = detect_gpu_type()

    if gpu_type and gpu_type in GPU_SPECS:
        gpu_spec = GPU_SPECS[gpu_type]
        peak_bf16_tflops = gpu_spec['bf16_tflops']
        peak_fp8_tflops = gpu_spec['fp8_tflops']

        print(f"  Current GPU: {gpu_spec['name']}")
        print(f"  Peak BF16:                 {peak_bf16_tflops} TFLOPS")
        print(f"  Peak FP8:                  {peak_fp8_tflops} TFLOPS")
        print(f"  Memory:                    {gpu_spec['memory_gb']} GB")
        print()

        # Show estimates for current GPU at different MFU levels
        peak_bf16 = peak_bf16_tflops * 1e12
        for mfu in [0.3, 0.4, 0.5]:
            effective_tflops = peak_bf16 * mfu
            time_per_step_ms = (flops['total_per_step'] / effective_tflops) * 1000
            tokens_per_sec = (batch_size * seq_len) / (time_per_step_ms / 1000)
            print(f"  At {mfu*100:.0f}% MFU on {gpu_type}:")
            print(f"    Time per step:           {time_per_step_ms:>12.2f} ms")
            print(f"    Throughput:              {tokens_per_sec:>12.0f} tokens/sec")
        print()

        # Show comparison with other GPUs
        print(f"  {'Comparison across GPU types:':-^76}")
        print(f"  {'GPU':<25} {'Peak BF16':<15} {'Time/step @ 40% MFU':<20} {'Throughput'}")
        print(f"  {'-'*25} {'-'*15} {'-'*20} {'-'*15}")

        for gpu_key in ['H100', 'H200', 'GB200', 'GB300']:
            spec = GPU_SPECS[gpu_key]
            peak = spec['bf16_tflops'] * 1e12
            effective = peak * 0.4  # 40% MFU
            time_ms = (flops['total_per_step'] / effective) * 1000
            throughput = (batch_size * seq_len) / (time_ms / 1000)
            current = " (current)" if gpu_key == gpu_type else ""
            print(f"  {spec['name'][:24]:<25}{current:<8} {spec['bf16_tflops']:>6} TFLOPS   {time_ms:>8.2f} ms         {throughput:>10.0f} tok/s")

    else:
        # Fallback if GPU not detected
        print(f"  GPU not detected or unknown. Showing H100 estimates.")
        h100_peak_bf16 = 989e12

        print(f"  H100 Peak (BF16):          989 TFLOPS")

        for mfu in [0.3, 0.4, 0.5]:
            effective_tflops = h100_peak_bf16 * mfu
            time_per_step_ms = (flops['total_per_step'] / effective_tflops) * 1000
            tokens_per_sec = (batch_size * seq_len) / (time_per_step_ms / 1000)
            print(f"  At {mfu*100:.0f}% MFU on H100:")
            print(f"    Time per step:           {time_per_step_ms:>12.2f} ms")
            print(f"    Throughput:              {tokens_per_sec:>12.0f} tokens/sec")

    print("=" * 80 + "\n")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
