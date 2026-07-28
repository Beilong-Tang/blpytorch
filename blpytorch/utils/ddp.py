import os
from functools import wraps
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from typing import Tuple


def setup_ddp() -> Tuple[bool, int]:
    # If not launched with torchrun, just run normally.
    if "RANK" not in os.environ:
        return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ["RANK"])

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # torchrun
    dist.init_process_group(backend="nccl")

    return True, rank, torch.device(f"cuda:{local_rank}")


def ddp_cleanup(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    return wrapper


def ddp_model_wrapper(model, rank, world_size):
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[
                rank,
            ],
        )
    return model
