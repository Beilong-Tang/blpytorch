from omegaconf import OmegaConf
from hydra.utils import instantiate

def init_obj(config: str):
    return instantiate(OmegaConf.load(config))