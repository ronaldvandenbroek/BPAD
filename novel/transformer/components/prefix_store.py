import numpy as np

class PrefixStore():
    def __init__(self, num_attributes, case_start_length=1, use_prefix_errors=True):
        # For ablation studies, setting use_prefix_errors to False will only store the last event of each prefix
        self.use_prefix_errors = use_prefix_errors

        self.num_attributes = num_attributes
        # Include a start index as there might be a buffer start case
        self.case_start_index = case_start_length * num_attributes

        self.prefix_value_store = {}
        self.prefix_length_store = {}
        self.prefix_case_values = []

    def add_prefix(self, prefix, event, event_value):
        # Check if the prefix is already in the store
        current_prefix_key = prefix.tobytes()
        if current_prefix_key not in self.prefix_value_store:
            # Create a new prefix value array
            prefix_value = np.zeros_like(prefix)
            event_index_start = self.case_start_index
        else:
            if not self.use_prefix_errors:
                # Create a new prefix value array but do load the current prefix length
                prefix_value = np.zeros_like(prefix)
                event_index_start = self.prefix_length_store[current_prefix_key]
            else:
                # Load the existing prefix value array
                prefix_value = self.prefix_value_store[current_prefix_key].copy()
                event_index_start = self.prefix_length_store[current_prefix_key]

        event_index_end = event_index_start + self.num_attributes

        # Add the new event to the prefix
        prefix[event_index_start:event_index_end] = event
        new_prefix_key = prefix.tobytes()

        # Add the new event values to the prefix values
        prefix_value[event_index_start:event_index_end] = event_value
        # Store the new values or update the existing values
        self.prefix_value_store[new_prefix_key] = prefix_value
        self.prefix_length_store[new_prefix_key] = event_index_end

        # Save the target + prefix values corresponding to the current event
        self.prefix_case_values.append(prefix_value)

    def add_prefixes(self, prefixes, events, event_values):
        for prefix, event, event_value in zip(prefixes, events, event_values):
            self.add_prefix(prefix, event, event_value)

    def get_prefix_case_values(self):
        return np.array(self.prefix_case_values)
