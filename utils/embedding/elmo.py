import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder

class ProcessELMoEncoder():
    def __init__(self,
                 encoders, 
                 attribute_types, 
                 event_attribute_keys,
                 features,
                 vector_size=1024) -> None:
        
        if vector_size != 1024:
            raise 'Error ELMo only allows for vector_sizes of 1024'

        #self.encoders = encoders
        self.attribute_types = attribute_types
        self.event_attribute_keys = event_attribute_keys
        self.features = features

        self.vector_size = vector_size

        # Initialize the ELMo embedder with a pretrained model
        self.elmo = ElmoEmbedder()

    def flat_features_2d(self):
        features = []
        for feature in self.features:
            encoded_feature = []
            for attr_trace in feature:
                sentence = attr_trace

                elmo_embedding = self.elmo.embed_sentence(sentence)
                word_embeddings = torch.mean(torch.tensor(elmo_embedding), dim=0)

                # trace_attributes = []
                # for attr in attr_trace:
                #     if attr != 0:
                #         # (num_words, 1024)

                #         trace_attributes.append(np.array(fourier_encoding(attr, self.frequencies),dtype=np.float32))
                #     else:
                #         trace_attributes.append(self.zero_vector)
                encoded_feature.append(word_embeddings)     
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

        





