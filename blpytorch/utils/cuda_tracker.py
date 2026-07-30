import torch


class CUDAMemoryTracker:
    def __init__(self, device=None):
        self.device = (
            torch.cuda.current_device() if device is None else device
        )
        self.reset()

    def reset(self):
        self.count = 0
        self.used_sum = 0

    def update(self):
        """
        Record the current GPU memory usage.

        Returns:
            used (int): Current used memory in bytes.
            pct (float): Current memory usage percentage.
        """
        free, total = torch.cuda.mem_get_info(self.device)
        used = total - free

        self.count += 1
        self.used_sum += used

        return used, 100.0 * used / total

    def summary(self):
        free, total = torch.cuda.mem_get_info(self.device)
        current = total - free
        average = self.used_sum / self.count if self.count else current

        return {
            "current_gb": current / 1024**3,
            "current_pct": 100.0 * current / total,
            "average_gb": average / 1024**3,
            "average_pct": 100.0 * average / total,
        }

    def summary_str(self):
        s = self.summary()
        return (
            f"CUDA Memory | "
            f"current: {s['current_gb']:.2f} GB ({s['current_pct']:.1f}%), "
            f"average: {s['average_gb']:.2f} GB ({s['average_pct']:.1f}%)"
        )