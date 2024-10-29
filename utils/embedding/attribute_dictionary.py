# RCVDB: Dictionary only needs to keep track of the unique values of a single attribute
from collections import defaultdict

class AttributeDictionary():
    def __init__(self, start_index=3, max_size=100, start_symbol= '▶', end_symbol='■') -> None:
        self.max_size = max_size
        self.start_index = start_index
        self.encodings = defaultdict(str)
        self.encodings_inv = defaultdict(int)
        
        # Hardcoding start and end symbols to reserve the space
        # 0 is reserved for padding
        self.encodings[start_symbol] = 1
        self.encodings_inv[1] = start_symbol
        self.encodings[end_symbol] = 2
        self.encodings_inv[2] = end_symbol       

    def encode_list(self, list):
        for i, label in enumerate(list):
            list[i] = self.encode(label)
        return list

    def encode(self, label):
        if label in self.encodings.keys():
            return str(self.encodings[label])
        else:
            existing_keys = self.encodings_inv.keys()
            new_key = len(existing_keys) + self.start_index
            self.encodings[label] = new_key
            self.encodings_inv[new_key] = label
            return str(new_key)
        
    def decode(self, value):
        value = int(value)
        if value in self.encodings_inv.keys():
            return self.encodings_inv[value]
        else:
            return None
        
    def encoded_attributes(self):
        return [str(i) for i in map(str, self.encodings.values())]
    
    def largest_attribute(self):
        return max(self.encodings.values())
    
    # Buffer attributes are all encoded labels that are not part of a label_value mapping
    def buffer_attributes(self):
        current_size = len(self.encodings_inv.keys())
        return [str(i) for i in range(current_size + self.start_index, self.max_size + self.start_index)]
    
    def __str__(self):
        current_size = len(self.encodings_inv.keys())
        return f"AttributeDictionary (size={current_size}, indexes=[{self.start_index},{current_size + self.start_index}], reserved={self.max_size - self.start_index})"
    
    def __repr__(self):
        return self.__str__()