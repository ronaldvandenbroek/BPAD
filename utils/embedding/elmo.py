import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm 


class ProcessELMoEncoder():
    def __init__(self,
                 features,
                 vector_size=1024) -> None:
        
        if vector_size != 1024:
            raise 'Error ELMo only allows for vector_sizes of 1024'

        # #self.encoders = encoders
        # self.attribute_types = attribute_types
        # self.event_attribute_keys = event_attribute_keys
        print("Setup ELMO Encoding")
        self.features = np.array(features).astype(str)
        print("Converted Features to strings")

        self.vector_size = vector_size

        # Initialize the ELMo embedder with a pretrained model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        print("Loaded BERT tokenizer and model")

    def encode_attribute(self, attribute):
        inputs = self.tokenizer(attribute, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token embedding

    def flat_elmo_features_2d(self):
        print("Encoding cases with ELMO")
        features = []
        for feature in tqdm(self.features, desc="Features", leave=False):
            encoded_feature = []
            for attr_trace in tqdm(feature, desc="Attribute Traces", leave=False):
                trace_attributes = []
                for attr in tqdm(attr_trace, desc="Attributes", leave=False):
                    enc_attr = self.encode_attribute(attr)
    
                    trace_attributes.append(enc_attr)

                    #     if attr != 0:
                    #         # (num_words, 1024)

                    #         trace_attributes.append(np.array(fourier_encoding(attr, self.frequencies),dtype=np.float32))
                    #     else:
                    #         trace_attributes.append(self.zero_vector)
                encoded_feature.append(trace_attributes)     
            features.append(encoded_feature)

        features = np.array(features, dtype=np.float32)
        print(features.shape)
        transposed_features = np.transpose(features, (1, 2, 0, 3))
        print(transposed_features.shape)

        dim0, dim1, dim2, dim3 = transposed_features.shape
        transposed_features = np.reshape(transposed_features, (dim0, dim1, dim2 * dim3))#, order='C')
        print(transposed_features.shape)
        transposed_features = np.reshape(transposed_features, (dim0, dim1 * dim2 * dim3))#, order='C')
        print(transposed_features.shape)

        return transposed_features

        





