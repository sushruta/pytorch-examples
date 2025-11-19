"""
Test script to demonstrate training metrics output.
Simulates a few training steps to show metric logging.
"""

import torch
from models import MyModel, ModelConfig
from models.model_analysis import print_model_architecture, print_flops_analysis, format_time
import time


def simulate_training():
    """Simulate training to show metrics output."""

    # Create a small model for testing
    print("Creating test model...")
    config = ModelConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1376,
        num_experts=8,
        num_experts_per_tok=2,
        moe_layer_indices=[1, 3],
        max_position_embeddings=1024,
    )

    model = MyModel(config)

    # Print architecture
    train_config = {
        'training': {
            'batch_size': 4,
            'sequence_length': 512,
            'gradient_accumulation_steps': 2,
            'learning_rate': 3e-4,
            'precision': 'bf16',
            'use_torch_compile': False,
            'distributed': False,
        }
    }

    print_model_architecture(model, train_config)
    print_flops_analysis(config, seq_len=512, batch_size=4)

    # Simulate training steps
    print("\n" + "="*80)
    print("STARTING TRAINING SIMULATION")
    print("="*80 + "\n")

    batch_size = 4
    seq_len = 512
    tokens_per_batch = batch_size * seq_len

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    start_time = time.time()
    total_tokens = 0

    for step in range(30):  # Simulate 30 steps
        # Create dummy batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        step_start = time.time()
        logits, loss, aux_loss = model(input_ids, labels=input_ids)

        # Backward pass
        loss_total = loss + (aux_loss if aux_loss is not None else 0)
        loss_total.backward()
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - step_start
        total_tokens += tokens_per_batch

        # Log every 10 steps
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_step_time_ms = (elapsed / (step + 1)) * 1000
            tokens_per_sec = total_tokens / elapsed
            samples_per_sec = (step + 1) * batch_size / elapsed
            steps_per_sec = (step + 1) / elapsed

            print(f"\n{'='*80}")
            print(f"Step {step + 1}/30 | Epoch 0")
            print(f"{'-'*80}")
            print(f"  Loss:              {loss.item():.4f}")
            if aux_loss is not None:
                print(f"  Aux Loss:          {aux_loss.item():.4f}")
            print(f"  Learning Rate:     3.00e-04")
            print(f"{'-'*80}")
            print(f"  Throughput:        {tokens_per_sec:>10.0f} tokens/sec")
            print(f"  Samples/sec:       {samples_per_sec:>10.2f}")
            print(f"  Steps/sec:         {steps_per_sec:>10.2f}")
            print(f"  Step time:         {avg_step_time_ms:>10.2f} ms")
            print(f"  Total tokens:      {total_tokens:>10,}")
            print(f"{'-'*80}")
            print(f"  Time elapsed:      {format_time(elapsed)}")
            remaining_steps = 30 - (step + 1)
            est_remaining = remaining_steps * (elapsed / (step + 1))
            print(f"  Est. remaining:    {format_time(est_remaining)}")
            print(f"{'='*80}\n")

            start_time = time.time()
            total_tokens = 0

    print("\n" + "="*80)
    print("TRAINING SIMULATION COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("="*80)
    print("TRAINING METRICS DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates the metrics that will be printed during training.")
    print("It creates a small model and simulates training to show the output format.\n")

    simulate_training()

    print("\nIn actual training on GPUs, you'll see:")
    print("  - Detailed model architecture at startup")
    print("  - FLOPS analysis and throughput estimates")
    print("  - Training metrics every N steps (configurable)")
    print("  - GPU memory usage (when running on CUDA)")
    print("  - Time estimates and progress tracking")
