import time

import numpy as np

class RuntimeTracker():
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        self.batch_runtimes = []

    def start_iteration(self):
        self.start_time = time.time()

    def end_iteration(self):
        self.batch_runtimes.append(time.time() - self.start_time)

    def get_average_std_run_time(self):
        per_item_runtimes = [runtime / self.batch_size for runtime in self.batch_runtimes]
        average_runtime = np.mean(per_item_runtimes)
        std_runtime = np.std(per_item_runtimes)
        nr_items = len(self.batch_runtimes) * self.batch_size

        return {
            'average_runtime': average_runtime,
            'std_runtime': std_runtime,
            'nr_items': nr_items
        }