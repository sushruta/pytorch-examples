"""
Distributed training utilities for expert parallelism.
Handles process group creation and communication.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Tuple


# Global process groups
_EXPERT_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None


def setup_distributed(
    expert_parallel_size: int = 1,
    data_parallel_size: Optional[int] = None,
    backend: str = "nccl",
) -> Tuple[int, int, int]:
    """
    Initialize distributed training with expert and data parallelism.

    Args:
        expert_parallel_size: Number of GPUs to split experts across
        data_parallel_size: Number of GPUs for data parallelism (auto if None)
        backend: Communication backend (nccl for GPU, gloo for CPU)

    Returns:
        world_size: Total number of processes
        rank: Global rank of this process
        local_rank: Local rank on this node
    """
    # Initialize process group
    if not dist.is_initialized():
        # Get environment variables set by torchrun/torch.distributed.launch
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
        else:
            # Single GPU - no distributed setup needed
            return world_size, rank, local_rank

        # Set device
        torch.cuda.set_device(local_rank)
    else:
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()

    # Validate configuration
    if data_parallel_size is None:
        data_parallel_size = world_size // expert_parallel_size

    if expert_parallel_size * data_parallel_size != world_size:
        raise ValueError(
            f"expert_parallel_size ({expert_parallel_size}) * "
            f"data_parallel_size ({data_parallel_size}) must equal "
            f"world_size ({world_size})"
        )

    # Create process groups for expert parallelism
    global _EXPERT_PARALLEL_GROUP, _DATA_PARALLEL_GROUP

    # Expert parallel groups: ranks that share data but have different experts
    # Data parallel groups: ranks that have same experts but different data
    for dp_rank in range(data_parallel_size):
        ep_ranks = list(range(
            dp_rank * expert_parallel_size,
            (dp_rank + 1) * expert_parallel_size
        ))
        group = dist.new_group(ep_ranks)

        if rank in ep_ranks:
            _EXPERT_PARALLEL_GROUP = group

    # Create data parallel groups
    for ep_rank in range(expert_parallel_size):
        dp_ranks = list(range(ep_rank, world_size, expert_parallel_size))
        group = dist.new_group(dp_ranks)

        if rank in dp_ranks:
            _DATA_PARALLEL_GROUP = group

    # Print detailed setup info
    if rank == 0:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"\n{'='*80}")
        print(f"Distributed Training Setup")
        print(f"{'-'*80}")
        print(f"  Backend:              {backend}")
        print(f"  World size:           {world_size}")
        print(f"  GPUs available:       {num_gpus}")
        print(f"  Expert parallel size: {expert_parallel_size}")
        print(f"  Data parallel size:   {data_parallel_size}")
        if torch.cuda.is_available():
            print(f"  GPU name:             {torch.cuda.get_device_name(0)}")
        print(f"{'='*80}\n")

    return world_size, rank, local_rank


def get_expert_parallel_group():
    """Get the expert parallel process group."""
    return _EXPERT_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel process group."""
    return _DATA_PARALLEL_GROUP


def get_expert_parallel_rank() -> int:
    """Get rank within expert parallel group."""
    if _EXPERT_PARALLEL_GROUP is None:
        return 0
    return dist.get_rank(_EXPERT_PARALLEL_GROUP)


def get_expert_parallel_world_size() -> int:
    """Get size of expert parallel group."""
    if _EXPERT_PARALLEL_GROUP is None:
        return 1
    return dist.get_world_size(_EXPERT_PARALLEL_GROUP)


def get_data_parallel_rank() -> int:
    """Get rank within data parallel group."""
    if _DATA_PARALLEL_GROUP is None:
        return 0
    return dist.get_rank(_DATA_PARALLEL_GROUP)


def get_data_parallel_world_size() -> int:
    """Get size of data parallel group."""
    if _DATA_PARALLEL_GROUP is None:
        return 1
    return dist.get_world_size(_DATA_PARALLEL_GROUP)


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank_0(message: str):
    """Print message only from rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


class DistributedSampler:
    """
    Simple distributed sampler for data parallelism.
    Each rank gets a different subset of the data.
    """

    def __init__(self, dataset_size: int, shuffle: bool = True, seed: int = 0):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.seed = seed

        self.rank = get_data_parallel_rank()
        self.world_size = get_data_parallel_world_size()

        # Compute per-rank size
        self.num_samples = dataset_size // self.world_size
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # Generate indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))

        # Truncate to total_size
        indices = indices[:self.total_size]

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Update seed for shuffling."""
        self.seed = epoch


def all_reduce_grads(model: torch.nn.Module) -> float:
    """
    All-reduce gradients across data parallel group.
    Used for data parallelism.

    Returns:
        Communication time in seconds
    """
    if _DATA_PARALLEL_GROUP is None:
        return 0.0

    world_size = get_data_parallel_world_size()
    if world_size == 1:
        return 0.0

    # Time the communication
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    import time
    start_time = time.time()

    # Collect all gradients
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=_DATA_PARALLEL_GROUP)
            param.grad.div_(world_size)

    # Synchronize to get accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    comm_time = time.time() - start_time
    return comm_time
