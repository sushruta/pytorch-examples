"""
Training script for MoE model.
Supports distributed training with expert parallelism, BF16/FP8, and torch.compile.
"""

import argparse
import os
import time
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any

from models import MyModel, ModelConfig
from models.model_analysis import (
    print_model_architecture,
    print_flops_analysis,
    format_time,
)
from training.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_expert_parallel_group,
    get_data_parallel_group,
    all_reduce_grads,
    print_rank_0,
    DistributedSampler,
)


class DummyDataset(Dataset):
    """
    Dummy dataset for testing/demonstration.
    Replace with your actual dataset.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random tokens
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay."""
    train_config = config['training']
    lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']

    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases, layer norms, and embeddings
        if 'bias' in name or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer


def get_lr_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """Create learning rate scheduler with warmup and cosine decay."""
    train_config = config['training']
    warmup_steps = train_config['warmup_steps']
    max_steps = train_config['max_steps']
    lr = train_config['learning_rate']
    min_lr = lr * 0.1  # Decay to 10% of max LR

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    accumulation_steps: int,
    step: int,
) -> Dict[str, float]:
    """
    Perform one training step.

    Args:
        model: The model to train
        batch: Dictionary with input_ids and labels
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        accumulation_steps: Number of gradient accumulation steps
        step: Current training step

    Returns:
        Dictionary with loss metrics
    """
    is_accumulating = (step + 1) % accumulation_steps != 0

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass with AMP
    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss, aux_loss = model(input_ids, labels=labels)

            # Add auxiliary loss
            if aux_loss is not None:
                total_loss = loss + aux_loss
            else:
                total_loss = loss

            # Scale loss for gradient accumulation
            total_loss = total_loss / accumulation_steps
    else:
        logits, loss, aux_loss = model(input_ids, labels=labels)
        if aux_loss is not None:
            total_loss = loss + aux_loss
        else:
            total_loss = loss
        total_loss = total_loss / accumulation_steps

    # Backward pass
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

    # Optimizer step (only on last accumulation step)
    comm_time = 0.0
    if not is_accumulating:
        # Gradient clipping
        if scaler is not None:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # All-reduce gradients for data parallelism (returns comm time)
        comm_time = all_reduce_grads(model)

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Return metrics
    metrics = {
        'loss': loss.item() if loss is not None else 0.0,
        'aux_loss': aux_loss.item() if aux_loss is not None else 0.0,
        'lr': scheduler.get_last_lr()[0],
        'comm_time': comm_time,
    }

    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Dict[str, Any],
    device: torch.device,
    rank: int,
):
    """Main training loop."""
    train_config = config['training']
    max_steps = train_config['max_steps']
    log_interval = train_config['log_interval']
    accumulation_steps = train_config['gradient_accumulation_steps']
    use_amp = train_config['precision'] in ['bf16', 'fp16']

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and train_config['precision'] == 'fp16' else None

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    running_aux_loss = 0.0
    running_comm_time = 0.0
    start_time = time.time()
    epoch_start_time = time.time()
    total_tokens = 0

    # For throughput calculation
    batch_size = train_config['batch_size']
    seq_len = train_config['sequence_length']
    tokens_per_batch = batch_size * seq_len

    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # End of epoch
            epoch += 1
            epoch_time = time.time() - epoch_start_time
            print_rank_0(f"\n{'='*80}")
            print_rank_0(f"Epoch {epoch} completed in {format_time(epoch_time)}")
            print_rank_0(f"{'='*80}\n")

            # Reset iterator
            train_iter = iter(train_loader)
            batch = next(train_iter)
            epoch_start_time = time.time()

        # Training step
        step_start_time = time.time()
        metrics = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            accumulation_steps=accumulation_steps,
            step=step,
        )
        step_time = time.time() - step_start_time

        running_loss += metrics['loss']
        running_aux_loss += metrics['aux_loss']
        running_comm_time += metrics['comm_time']
        total_tokens += tokens_per_batch

        # Logging
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / log_interval
            avg_aux_loss = running_aux_loss / log_interval

            # Throughput metrics
            tokens_per_sec = (log_interval * tokens_per_batch) / elapsed
            samples_per_sec = (log_interval * batch_size) / elapsed
            steps_per_sec = log_interval / elapsed
            avg_step_time_ms = (elapsed / log_interval) * 1000

            # Communication metrics
            avg_comm_time_ms = (running_comm_time / log_interval) * 1000
            comm_percentage = (running_comm_time / elapsed) * 100 if elapsed > 0 else 0

            # Memory stats
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated(device) / 1e9
                memory_reserved = torch.cuda.max_memory_reserved(device) / 1e9
            else:
                memory_allocated = 0
                memory_reserved = 0

            # Print comprehensive metrics
            print_rank_0(f"\n{'='*80}")
            print_rank_0(f"Step {step + 1}/{max_steps} | Epoch {epoch}")
            print_rank_0(f"{'-'*80}")
            print_rank_0(f"  Loss:              {avg_loss:.4f}")
            print_rank_0(f"  Aux Loss:          {avg_aux_loss:.4f}")
            print_rank_0(f"  Learning Rate:     {metrics['lr']:.2e}")
            print_rank_0(f"{'-'*80}")
            print_rank_0(f"  Throughput:        {tokens_per_sec:>10.0f} tokens/sec")
            print_rank_0(f"  Samples/sec:       {samples_per_sec:>10.2f}")
            print_rank_0(f"  Steps/sec:         {steps_per_sec:>10.2f}")
            print_rank_0(f"  Step time:         {avg_step_time_ms:>10.2f} ms")
            print_rank_0(f"  Total tokens:      {total_tokens:>10,}")

            # Communication stats (only if distributed)
            if train_config['distributed'] and train_config['data_parallel_size'] > 1:
                print_rank_0(f"{'-'*80}")
                print_rank_0(f"  Communication:")
                print_rank_0(f"    NCCL time:       {avg_comm_time_ms:>10.2f} ms/step")
                print_rank_0(f"    Comm overhead:   {comm_percentage:>10.1f}%")

            if torch.cuda.is_available():
                print_rank_0(f"{'-'*80}")
                print_rank_0(f"  GPU Memory:")
                print_rank_0(f"    Allocated:       {memory_allocated:>10.2f} GB")
                print_rank_0(f"    Reserved:        {memory_reserved:>10.2f} GB")

            # Time remaining estimate
            steps_remaining = max_steps - (step + 1)
            time_per_step = elapsed / log_interval
            estimated_time_remaining = steps_remaining * time_per_step
            print_rank_0(f"{'-'*80}")
            print_rank_0(f"  Time elapsed:      {format_time(time.time() - start_time)}")
            print_rank_0(f"  Est. remaining:    {format_time(estimated_time_remaining)}")
            print_rank_0(f"{'='*80}\n")

            running_loss = 0.0
            running_aux_loss = 0.0
            running_comm_time = 0.0
            start_time = time.time()

        step += 1

    print_rank_0("\n" + "="*80)
    print_rank_0("Training completed!")
    print_rank_0("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train MoE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_config = config['model']
    train_config = config['training']

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup distributed training
    if train_config['distributed']:
        world_size, rank, local_rank = setup_distributed(
            expert_parallel_size=train_config['expert_parallel_size'],
            data_parallel_size=train_config['data_parallel_size'],
        )
        device = torch.device(f"cuda:{local_rank}")
    else:
        world_size, rank, local_rank = 1, 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print GPU info
    print_rank_0(f"\n{'='*80}")
    print_rank_0(f"Hardware Configuration")
    print_rank_0(f"{'-'*80}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print_rank_0(f"  Device:            {device}")
        print_rank_0(f"  GPUs available:    {num_gpus}")
        print_rank_0(f"  GPU name:          {torch.cuda.get_device_name(0)}")
        if train_config['distributed']:
            print_rank_0(f"  World size:        {world_size}")
            print_rank_0(f"  Rank:              {rank}")
            print_rank_0(f"  Local rank:        {local_rank}")
    else:
        print_rank_0(f"  Device:            {device}")
    print_rank_0(f"{'='*80}\n")

    # Create model config with expert parallel group
    config_obj = ModelConfig(**model_config)
    if train_config['distributed']:
        config_obj.expert_parallel_group = get_expert_parallel_group()

    # Create model
    print_rank_0("Creating model...")
    model = MyModel(config_obj)
    model = model.to(device)

    # Print detailed architecture analysis
    if rank == 0:
        print_model_architecture(model, config)
        print_flops_analysis(
            config_obj,
            seq_len=train_config['sequence_length'],
            batch_size=train_config['batch_size']
        )

    # Apply torch.compile if enabled
    if train_config.get('use_torch_compile', False) and hasattr(torch, 'compile'):
        print_rank_0("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create dataset and dataloader
    print_rank_0("Creating dataset...")
    dataset = DummyDataset(
        num_samples=100000,
        seq_len=train_config['sequence_length'],
        vocab_size=model_config['vocab_size'],
    )

    # Use distributed sampler for data parallelism
    if train_config['distributed'] and train_config['data_parallel_size'] > 1:
        sampler = DistributedSampler(len(dataset), shuffle=True, seed=args.seed)
        train_loader = DataLoader(
            dataset,
            batch_size=train_config['batch_size'],
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    # Create optimizer and scheduler
    print_rank_0("Creating optimizer...")
    optimizer = create_optimizer(model, config)
    scheduler = get_lr_scheduler(optimizer, config)

    # Train
    print_rank_0("Starting training...")
    train(model, train_loader, optimizer, scheduler, config, device, rank)

    # Cleanup
    if train_config['distributed']:
        cleanup_distributed()


if __name__ == "__main__":
    main()
