from .seed import setup_seed
from .ddp import setup_ddp, ddp_model_wrapper, ddp_cleanup
from .log import setup_logger, Logger
from .init import init_obj
from .config import dump_config, load_config

from typing import Tuple
import os
import torch

def setup_all_ddp(seed, logdir, init_process=True) -> Tuple[bool, int, Logger]:
    """Set up distributed training, random seed, and logger.

    Args:
        seed: Base random seed.
        logdir: Directory for log files.
        init_process: whther to run dist.init_process (usually disabled during sampling where our goal is just run parallel computing)

    Returns:
        A tuple containing:
            - is_ddp: Whether distributed training is enabled.
            - rank: Global process rank.
            - logger: Configured logger.
    """
    is_ddp, rank, device = setup_ddp(init_process=init_process)
    seed = setup_seed(seed, rank)
    logger = setup_logger(logdir, rank)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    return is_ddp, rank, logger, world_size, device


def setup_all_parallel(logdir, seed):
    """Set up parallel process based on the rank and os.environ WORLD_SIZE
    
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger = setup_logger(logdir, rank)
    seed = setup_seed(seed, rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else 'cpu')
    return rank, world_size, logger, device