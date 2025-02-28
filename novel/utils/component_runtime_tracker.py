import time


class ComponentRuntimeTracker():
    def __init__(self):     
        self.component_start_times = {}
        self.component_end_times = {}
        self.component_final_times = {}

    def start_component(self, component_name):
        self.component_start_times[component_name] = time.time()

    def end_component(self, component_name):
        self.component_end_times[component_name] = time.time()

        # Add the component time to the final time
        if component_name not in self.component_final_times:
            self.component_final_times[component_name] = 0
            
        self.component_final_times[component_name] += self.component_end_times[component_name] - self.component_start_times[component_name]
