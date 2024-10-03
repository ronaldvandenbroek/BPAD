from collections import defaultdict
from gensim.models import Word2Vec
import numpy as np
import copy

# RCVDB: Dictionary only needs to keep track of the unique values of a single attribute
class AttributeDictionary():
    def __init__(self, start_index=1, max_size=100) -> None:
        self.max_size = max_size
        self.start_index = start_index
        self.encodings = defaultdict(str)
        self.encodings_inv = defaultdict(int)

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
        return [str(i) for i in map(str, self.encodings_inv.keys())]
    
    # Buffer attributes are all encoded labels that are not part of a label_value mapping
    def buffer_attributes(self):
        current_size = len(self.encodings_inv.keys())
        return [str(i) for i in range(current_size + self.start_index, self.max_size + self.start_index)]
    
    def __str__(self):
        current_size = len(self.encodings_inv.keys())
        return f"AttributeDictionary (size={current_size}, indexes=[{self.start_index},{current_size + self.start_index}], reserved={self.max_size + self.start_index})"
    
    def __repr__(self):
        return self.__str__()

class ProcessWord2Vec():
    def __init__(self, training_sentences, vector_size=50, window=5, min_count=1, workers=4, attr_dicts=None, debug=False) -> None:
        self.debug = debug
        self.attr_dicts = attr_dicts

        if self.attr_dicts is not None:
            training_sentences = self._encode_training_sentences(copy.deepcopy(training_sentences))

        self.w2v_model = Word2Vec(sentences=training_sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    def encode_attribute(self, input):
        return np.array(self.w2v_model.wv[int(input)])

    # def _encode_training_sentences(self, training_sentances):
    #     if self.debug: print(f"Training Sentances: {training_sentances}")

    #     for i, sentance in enumerate(training_sentances):
    #         for j, word in enumerate(sentance):
    #             encoded_word = self.attr_dicts[j].encode(word)
    #             training_sentances[i][j] = encoded_word

    #     for attr_dict in self.attr_dicts:
    #         attr_dict:AttributeDictionary
    #         training_sentances += ProcessWord2Vec.convert_to_sentences(attr_dict.buffer_attributes())
    #     training_sentances

    #     if self.debug: print(f"Encoded Training Sentances: {training_sentances}")
    #     return training_sentances         

    # # Function to get the Word2Vec vector for an attribute
    # def _get_attr_vector(self, attr_index, attr):
    #     if self.attr_dicts is not None:
    #         mapped_attr = self.attr_dicts[attr_index].encode(attr)
            
    #         if self.debug: print(f"\t\tMapped {attr} to {mapped_attr}")
    #     else:
    #         mapped_attr = attr
    #     return self.w2v_model.wv[mapped_attr]

    # # Function to encode an event by averaging its attribute vectors
    # def _encode_event(self, event):
    #     if self.debug: print(f"\tEncoding Event: {event}")
    #     attribute_vectors = np.array([self._get_attr_vector(i, attr) for i, attr in enumerate(event)])
    #     event_vector = np.mean(attribute_vectors, axis=0)
    #     return event_vector

    # # Function to encode a trace by concatenating its event vectors
    # def encode_trace(self, trace):
    #     if self.debug: print(f"Encoding Trace: {trace}")
    #     event_vectors = [self._encode_event(event) for event in trace]
    #     trace_vector = np.concatenate(event_vectors, axis=0)
    #     return trace_vector      
    
    # @staticmethod
    # def convert_to_sentences(input):
    #     return [[str(i)] for i in input]