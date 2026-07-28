class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def update(self, **kwargs):
        self.count += 1
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v / self.count))
        return dict(avg_stats)

    def state_dict(self):
        return {"count": self.count, "stats": self.stats}

    def load_state_dict(self, state_dict):
        self.count = state_dict['count']
        self.stats = state_dict['stats']

    def log(self, digit = 3):
        avg_stats = self.extract()
        return " ".join([f"{k}: {v:.{digit}f}" for k, v in avg_stats.items()])