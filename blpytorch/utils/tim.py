from contextlib import contextmanager
import time
from datetime import timedelta
from contextlib import contextmanager

def time_to_str(time):
    return str(timedelta(seconds=int(time)))

from collections import deque
import time
from datetime import timedelta
from contextlib import contextmanager


class ContextTimer:
    """Flexible timer with context manager support for multiple processes."""

    def __init__(self, total_steps, disable=True, avg_window=10):
        self.total_steps = total_steps
        self.disable = disable

        self.step_count = 0
        self.timers = {}

        self.step_start_time = time.time()

        # Rolling average
        self.avg_window = avg_window
        self.step_history = deque(maxlen=avg_window)
        self.total_step_time=0.0

    @contextmanager
    def track(self, key):
        """Use like `with timer.track("data"):`"""
        if self.disable:
            yield
            return

        t0 = time.time()
        yield
        elapsed = time.time() - t0
        if key not in self.timers:
            self.timers[key] = deque(maxlen=self.avg_window)
        self.timers[key].append(elapsed)

    def update(self):
        elapsed = time.time() - self.step_start_time
        self.total_step_time += elapsed

        self.step_count += 1

        # Update rolling window
        self.step_history.append(elapsed)

        self.step_start_time = time.time()
        

    def stats(self):
        if self.step_count == 0:
            step_times = {}
        else:
            step_times = {
                k: sum(t) / len(t)
                for k, t in self.timers.items()
                if len(t) > 0
            }

        # ETA uses rolling average instead of global average
        instant_velocity = sum(self.step_history) / len(self.step_history) if self.step_history else 0.0
        eta_step = instant_velocity if self.step_history else 0.0
        eta = (
            time_to_str(int((self.total_steps - self.step_count) * eta_step))
            if eta_step > 0
            else time_to_str(0)
        )

        res = (
            f"[{time_to_str(self.total_step_time)}<{eta}, "
            f"{eta_step:.3f}s/it"
        )

        for k, v in step_times.items():
            res += f", {k}: {v:.3f}s"

        res += "]"
        return res

if __name__ == "__main__":
    total_steps = 100
    tim = ContextTimer(total_steps=100)

    for i in range(0, 100):
        tim.step_start()
        time.sleep(0.1)
        print(tim.stats())
        tim.step_end()
        pass

