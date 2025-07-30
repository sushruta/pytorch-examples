import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from dataset import FakeGPTDataset
from workload import BaseWorkloadConfig


def create_optimized_dataloader(wconf: BaseWorkloadConfig, world_size: int, rank: int):
    dataset = FakeGPTDataset(wconf)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=wconf.per_device_batch_size,
        sampler=sampler,
        num_workers=wconf.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler


def create_dummy_dataloader(wconf: BaseWorkloadConfig, sampler: DistributedSampler):
    if dist.get_rank() == 0:
        print("Creating synthetic data")
    # NOTE: we are not mentioning device!
    # the distributed sampler will take care of sending data to the devices
    inputs = torch.randint(
        1, wconf.vocab_size, (wconf.num_samples, wconf.seq_len), dtype=torch.long
    )
    labels = torch.cat(
        (inputs[:, 1:], torch.zeros((wconf.num_samples, 1), dtype=torch.long)), dim=-1
    )
    dataset = TensorDataset(inputs, labels)

    return DataLoader(
        dataset,
        batch_size=wconf.per_device_batch_size,
        sampler=sampler,
        num_workers=wconf.dataloader_num_workers,
    )
