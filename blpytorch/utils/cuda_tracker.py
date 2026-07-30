import torch


import torch


class CUDAMemoryTracker:
    def __init__(self, device=None):
        if device is None:
            device = torch.cuda.current_device()

        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.count = 0
        self.sum_used = 0
        self.min_used = float("inf")
        self.max_used = 0

    def update(self):
        """Record the current GPU memory usage."""
        free, total = torch.cuda.mem_get_info(self.device)
        used = total - free

        self.count += 1
        self.sum_used += used
        self.min_used = min(self.min_used, used)
        self.max_used = max(self.max_used, used)

        return used

    @property
    def total_memory(self):
        return torch.cuda.get_device_properties(self.device).total_memory

    @property
    def avg_used(self):
        return self.sum_used / self.count if self.count else 0

    def summary(self):
        total = self.total_memory

        return {
            "avg_used_gb": self.avg_used / 1024**3,
            "min_used_gb": self.min_used / 1024**3,
            "max_used_gb": self.max_used / 1024**3,
            "avg_pct": 100 * self.avg_used / total,
            "min_pct": 100 * self.min_used / total,
            "max_pct": 100 * self.max_used / total,
        }

    def summary_str(self):
        s = self.summary()
        return (
            f"CUDA Memory | "
            f"avg: {s['avg_used_gb']:.2f} GB ({s['avg_pct']:.1f}%), "
            f"min: {s['min_used_gb']:.2f} GB ({s['min_pct']:.1f}%), "
            f"max: {s['max_used_gb']:.2f} GB ({s['max_pct']:.1f}%)"
        )