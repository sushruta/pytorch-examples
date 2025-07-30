import torch
from torch.utils.data import Dataset
from workload import BaseWorkloadConfig


class FakeGPTDataset(Dataset):
    def __init__(self, wconf: BaseWorkloadConfig):
        self.vocab_size = wconf.vocab_size
        self.seq_len = wconf.seq_len
        self.num_samples = wconf.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.randint(1, self.vocab_size, (self.seq_len,), dtype=torch.long)
        labels = torch.cat((inputs[1:], torch.tensor([0], dtype=torch.long)))
        return inputs, labels
