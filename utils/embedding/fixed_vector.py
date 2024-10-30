import numpy as np

from utils.embedding.util import fourier_encoding

class FixedVectorEncoder():
    def __init__(self, vector_size, features):
        self.vector_size = vector_size
        self.features = features

        # Precompute
        self.frequencies = np.linspace(1, vector_size / 2, vector_size // 2)
        self.zero_vector = np.zeros(vector_size, dtype=np.float32)

    def flat_features_2d(self):
        features = []
        for feature in self.features:
            encoded_feature = []
            for attr_trace in feature:
                trace_attributes = []
                for attr in attr_trace:
                    if attr != 0:
                        trace_attributes.append(np.array(fourier_encoding(attr, self.frequencies),dtype=np.float32))
                    else:
                        trace_attributes.append(self.zero_vector)
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