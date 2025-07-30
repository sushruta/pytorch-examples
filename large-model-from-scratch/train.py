import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
import argparse
import dataclasses
from contextlib import nullcontext
from rich.console import Console
from rich.table import Table
from model import SimpleGPT
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from dataloader import create_optimized_dataloader
from env import cleanup_distributed, setup_distributed
from profiler import CombinedProfiler
from workload import (
    BaseWorkloadConfig,
    WorkloadConfig1B,
    WorkloadConfig2B,
    WorkloadConfig4B,
    WorkloadConfig8B,
    WorkloadConfig16B,
    WorkloadConfig32B,
    WorkloadConfig70B,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found!")
    print("training will use CUDA")

    parser = argparse.ArgumentParser(description="FSDP training script for large models.")
    parser.add_argument(
        "--model_size",
        type=str,
        default="16B",
        choices=["cz", "1B", "2B", "4B", "8B", "16B", "32B", "70B"],
        help="Model size to train. Defaults to 16B.",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Per device batch size for training. Defaults to 8.",
    )
    args, _ = parser.parse_known_args()

    # rank is global rank
    local_rank, rank, world_size = setup_distributed()

    config_map = {
        "cz": BaseWorkloadConfig,
        "1B": WorkloadConfig1B,
        "2B": WorkloadConfig2B,
        "4B": WorkloadConfig4B,
        "8B": WorkloadConfig8B,
        "16B": WorkloadConfig16B,
        "32B": WorkloadConfig32B,
        "70B": WorkloadConfig70B,
    }
    wconf_class = config_map.get(args.model_size)
    if wconf_class is None:
        raise ValueError(f"Invalid model size {args.model_size}")
    wconf = wconf_class()

    # Override per_device_batch_size with the command-line argument
    wconf.per_device_batch_size = args.per_device_batch_size

    if rank == 0:
        console = Console()
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title=f"Workload Configuration ({args.model_size})",
        )
        table.add_column("Parameter", justify="left", style="dim")
        table.add_column("Value", justify="left")

        for field in dataclasses.fields(wconf):
            table.add_row(field.name, str(getattr(wconf, field.name)))

        table.add_row("world_size", str(world_size))
        console.print(table)

    model = SimpleGPT(wconf)
    model_num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model initialized with {model_num_params / 1e9:.2f}B parameters")

    # the model is casted into FSDP strategy
    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        cpu_offload=None,
    ).to(local_rank)

    # defining the optimizer, etc.
    optimizer = torch.optim.AdamW(model.parameters(), lr=wconf.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_dataloader, train_sampler = create_optimized_dataloader(wconf, world_size, rank)

    # initialize the profiler
    # ib_profiler = IBProfiler(model)
    profiler = CombinedProfiler(model)

    if rank == 0:
        print(f"Starting training using FSDP with {world_size} GPUs")

    if dist.is_initialized():
        dist.barrier()

    global_step = 0
    for epoch in range(wconf.max_epochs):
        # this will ensure our data is shuffled differently for each batch
        train_sampler.set_epoch(epoch)
        model.train()

        # get a nice looking progress bar
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{wconf.max_epochs}", disable=(rank != 0)
        )

        accum = wconf.gradient_accumulation_steps
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            # ib_profiler.step_start(global_step, rank)

            profiler.step_start(global_step, rank)
            inputs, labels = (
                inputs.to(local_rank, non_blocking=True),
                labels.to(local_rank, non_blocking=True),
            )

            ctx = model.no_sync() if (batch_idx % accum) != accum - 1 else nullcontext()

            with ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()

            if (batch_idx + 1) % wconf.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

            # ib_profiler.step_end(global_step, rank)
            profiler.step_end(global_step, rank)
            global_step += 1

    cleanup_distributed()

    print("training finished")


if __name__ == "__main__":
    main()
