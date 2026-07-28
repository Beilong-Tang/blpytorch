import torch
import random
import numpy as np

def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    return SEED