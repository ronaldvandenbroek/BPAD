import numpy as np
from tqdm import tqdm

from utils.embedding.util import fourier_encoding, recalculate_attribute_dimensions
from utils.enums import AttributeType

class FixedVectorEncoder():
    def __init__(self, 
                 vector_size, 
                 features,
                 attribute_types, 
                 event_attribute_keys):
        self.vector_size = vector_size
        self.features = features
        self.attribute_types = attribute_types
        self.event_attribute_keys = event_attribute_keys

        # Precompute
        self.frequencies = np.linspace(1, vector_size / 2, vector_size // 2)
        self.zero_vector = np.zeros(vector_size, dtype=np.float32)

    def encode_features(self, match_numerical=False):
        numeric_features = []
        numeric_feature_names = []
        categorical_features = []
        categorical_feature_names = []

        for index, (feature, attribute_type, attribute_key) in tqdm(enumerate(zip(self.features, self.attribute_types, self.event_attribute_keys)), "Encoding Fixed Features"):
            if attribute_type == AttributeType.NUMERICAL:
                if match_numerical:
                    encoded_feature = []
                    for attr_trace in feature:
                        trace_attributes = []
                        for attr in attr_trace:
                            if attr != 0:
                                trace_attributes.append(np.array(fourier_encoding(attr, self.frequencies),dtype=np.float32))
                            else:
                                trace_attributes.append(self.zero_vector)
                        encoded_feature.append(trace_attributes)     
                    numeric_features.append(np.array(encoded_feature, dtype=np.float32))
                else:
                    numeric_features.append(np.array(feature,dtype=np.float32))
                numeric_feature_names.append(attribute_key)

            elif attribute_type == AttributeType.CATEGORICAL:
                encoded_feature = []
                for attr_trace in feature:
                    trace_attributes = []
                    for attr in attr_trace:
                        if attr != 0:
                            trace_attributes.append(np.array(fourier_encoding(attr, self.frequencies),dtype=np.float32))
                        else:
                            trace_attributes.append(self.zero_vector)
                    encoded_feature.append(trace_attributes)
                categorical_features.append(np.array(encoded_feature, dtype=np.float32))
                categorical_feature_names.append(attribute_key)

        return np.array(categorical_features, dtype=np.float32), np.array(numeric_features, dtype=np.float32), np.array(numeric_feature_names), np.array(categorical_feature_names)

    def flat_features_2d(self, match_numerical=False):
        categorical_features, numeric_features, numeric_feature_names, categorical_feature_names = self.encode_features(match_numerical=match_numerical)

        transposed_categorical_features = np.transpose(categorical_features, (1, 2, 0, 3))
        transposed_numeric_features = np.transpose(numeric_features, (1, 2, 0))

        reordered_slices = []
        numeric_feature_names = list(numeric_feature_names)
        categorical_feature_names = list(categorical_feature_names)
        # Iterate through each feature in the total_order
        for feature in self.event_attribute_keys:
            if feature in numeric_feature_names:
                feature_index = numeric_feature_names.index(feature)
                if match_numerical:
                    reordered_slices.append(transposed_numeric_features[:, :, feature_index:feature_index+1])  # Retain shape (35483, 13, 1)
                else:
                    reordered_slices.append(transposed_categorical_features[:, :, feature_index, :])  # Shape (35483, 13, 200)
            elif feature in categorical_feature_names:
                feature_index = categorical_feature_names.index(feature)
                reordered_slices.append(transposed_categorical_features[:, :, feature_index, :])  # Shape (35483, 13, 200)

        # Concatenate all slices along the last axis
        merged_features = np.concatenate(reordered_slices, axis=-1)

        print(merged_features.shape)
        dim0, dim1, dim2 = merged_features.shape
        flat_merged_features = np.reshape(merged_features, (dim0, dim1 * dim2))#, order='C')
        print(flat_merged_features.shape)

        return flat_merged_features, recalculate_attribute_dimensions(self.attribute_types, self.vector_size, sort=False, match_numerical=False)