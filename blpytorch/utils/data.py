import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

def split_dataset(dataset: Dataset, world_size : int, rank: int) -> Dataset:
    """
    world_size: how many splits 
    rank: idx of splits which this process will take
    """
    generator = torch.Generator().manual_seed(1234)
    num_samples = len(dataset)
    split_size = num_samples // world_size
    remainder = num_samples % world_size
    split_sizes = [split_size] * world_size
    for i in range(remainder):
        split_sizes[i] += 1
    splits = random_split(dataset, split_sizes, generator=generator)
    return splits[rank]
