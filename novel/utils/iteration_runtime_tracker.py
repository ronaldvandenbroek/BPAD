import time

import numpy as np

class IterationRuntimeTracker():
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
    
    @staticmethod
    def sequentially_merge_runtimes(runtime_results):
            mean_overall_runtime_results = 0
            mean_inference_runtime_results = 0
            variance_overall_runtime_results = 0
            variance_inference_runtime_results = 0
            nr_items = runtime_results[0]['overall_runtimes']['nr_items']
            for runtime_result in runtime_results:
                mean_overall_runtime_results += runtime_result['overall_runtimes']['average_runtime']
                mean_inference_runtime_results += runtime_result['inference_runtimes']['average_runtime']
                variance_overall_runtime_results += runtime_result['overall_runtimes']['std_runtime'] ** 2
                variance_inference_runtime_results += runtime_result['inference_runtimes']['std_runtime'] ** 2

            std_overall_runtime_results = np.sqrt(variance_overall_runtime_results)
            std_inference_runtime_results = np.sqrt(variance_inference_runtime_results)

            return {
                'overall_runtimes': {
                    'average_runtime': mean_overall_runtime_results,
                    'std_runtime': std_overall_runtime_results,
                    'nr_items': nr_items
                },
                'inference_runtimes': {
                    'average_runtime': mean_inference_runtime_results,
                    'std_runtime': std_inference_runtime_results,
                    'nr_items': nr_items
                }
            }