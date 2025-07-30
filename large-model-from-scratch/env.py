import os

import torch
import torch.distributed as dist


def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    dist.destroy_process_group()
