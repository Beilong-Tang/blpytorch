import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
)

# Initialize NVML once
nvmlInit()

def get_gpu_stats(device=None):
    """
    Get GPU utilization and memory usage.

    Args:
        device (int, optional): GPU index. If None, uses the current PyTorch device.

    Returns:
        gpu_util (int): GPU utilization (%)
        mem_used_gb (float): Memory used (GiB)
        mem_total_gb (float): Total memory (GiB)
    """
    if device is None:
        device = torch.cuda.current_device()

    handle = nvmlDeviceGetHandleByIndex(device)
    util = nvmlDeviceGetUtilizationRates(handle)
    mem = nvmlDeviceGetMemoryInfo(handle)

    mem_pct = 100 * mem.used / mem.total
    return (
        f"[GPU Util: {util.gpu}% | Mem: {mem_pct}%]"
    )
