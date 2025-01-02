from gensim.models import Word2Vec

import numpy as np
import tensorflow as tf
from keras.layers import Layer, Embedding

from utils.embedding.attribute_dictionary import AttributeDictionary

# https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/

class TransformerWord2VecEncoder(Layer):
    def __init__(self, attribute_keys, sequence_length, encoders, dim_model=50, event_encoding=True, **kwargs) -> None:
        super(TransformerWord2VecEncoder, self).__init__(**kwargs)

        self.dim_model = dim_model
        self.sequence_length = sequence_length

        self.num_attributes = len(attribute_keys)
        case_length = sequence_length / self.num_attributes
        self.case_length = tf.cast(case_length, tf.int32)
        print("case_length", self.case_length)

        # Attribute Encoding
        attribute_key_dict, attribute_keys_tensor = TransformerWord2VecEncoder._convert_encoder_keys(encoders)

        self.attribute_key_trace_mask, self.categorical_mask, self.numerical_mask, self.attribute_mask = TransformerWord2VecEncoder._create_attribute_masks(
            attribute_keys, attribute_key_dict, self.case_length)

        extracted_w2v_models = TransformerWord2VecEncoder._create_w2v_models(encoders, dim_model)
        self.lookup_tables = TransformerWord2VecEncoder._create_lookup_tables(attribute_keys_tensor, extracted_w2v_models)

        # Positional Encoding
        if event_encoding:
            # If event encoding, create a positional matrix for each event and repeat it for each attribute 
            # so that each attribute has the same positional encoding if they share the same event
            positional_matrix_event = TransformerWord2VecEncoder._positional_encoding(self.case_length, dim_model)
            self.positional_matrix_trace = tf.repeat(positional_matrix_event, self.num_attributes, axis=1)
        else:
            # Else every attribute in the trace sequence has its own positional encoding
            self.positional_matrix_trace = TransformerWord2VecEncoder._positional_encoding(sequence_length, dim_model)

        print("positional_matrix_trace", self.positional_matrix_trace.shape)

    @staticmethod
    def _convert_encoder_keys(encoders):
        print("encoders", encoders.keys())
        attribute_key_dict = {}
        for i, key in enumerate(encoders.keys()):
            attribute_key_dict[key] = i
        print("attribute_key_dict", attribute_key_dict)

        attribute_keys_tensor = tf.constant(list(attribute_key_dict.values()), dtype=tf.int32)
        print("attribute_keys_tensor", attribute_keys_tensor)

        return attribute_key_dict, attribute_keys_tensor

    @staticmethod
    def _create_attribute_masks(attribute_keys, attribute_key_dict, case_length):
        # Convert attribute keys to integers based on the encoders
        attribute_keys_event_mask = []
        attribute_categorical_mask = []
        for attribute_key in attribute_keys:
            if attribute_key not in attribute_key_dict:
                attribute_keys_event_mask.append(-1) # -1 if numerical
                attribute_categorical_mask.append(False)
            else:
                attribute_keys_event_mask.append(attribute_key_dict[attribute_key])
                attribute_categorical_mask.append(True)

        attribute_keys_event_mask = tf.constant(attribute_keys_event_mask, dtype=tf.int32)
        print("attribute_keys_event_mask", attribute_keys_event_mask)

        # Expand the attribute keys to the sequence length
        attribute_key_trace_mask = tf.tile(attribute_keys_event_mask, multiples=[case_length])
        categorical_mask = tf.tile(attribute_categorical_mask, multiples=[case_length])
        numerical_mask = tf.logical_not(categorical_mask)
        print("attribute_key_trace_mask shape", attribute_key_trace_mask.shape, "trace_categorical_mask", categorical_mask.shape)
        # print("attribute_key_trace_mask", self.attribute_key_trace_mask)
        # print("trace_categorical_mask", self.categorical_mask)

        return attribute_key_trace_mask, categorical_mask, numerical_mask, attribute_keys_event_mask

    @staticmethod
    def _convert_to_sentences(input):
        return [[str(i)] for i in input]

    @staticmethod
    def _create_w2v_models(encoders, dim_model):
        # For each categorical attribute, create a Word2Vec model
        w2v_models = {}
        for i, encoder in enumerate(encoders.values()):
            encoder:AttributeDictionary

            print(encoder.encoded_attributes())
            w2v_model:Word2Vec = Word2Vec(
                sentences=TransformerWord2VecEncoder._convert_to_sentences(range(encoder.max_size + 1)),
                vector_size=dim_model,
                window=5, min_count=1, workers=4, sg=1, hs=0, negative=0)
            w2v_models[i] = w2v_model

        print(w2v_models.keys())
        for w2v_model in w2v_models.values():
            print(w2v_model.wv)

        extracted_w2v_models = {}
        for i, w2v_model in enumerate(w2v_models.values()):
            extracted_w2v_model = {}
            for word in w2v_model.wv.index_to_key:
                extracted_w2v_model[int(word)] = w2v_model.wv.get_vector(word)
            extracted_w2v_models[i] = extracted_w2v_model

        print("extracted_w2v_models")
        print(extracted_w2v_models.keys())
        for key, value in extracted_w2v_models[1].items():
            print(key, value.shape)

        return extracted_w2v_models       

    @staticmethod
    def _create_lookup_tables(attribute_keys_tensor, extracted_w2v_models):
        lookup_tables = {}
        for model_index, model_key in enumerate(attribute_keys_tensor):
            extracted_w2v_model = extracted_w2v_models[model_index]

            words = list(extracted_w2v_model.keys())
            vectors = [extracted_w2v_model[word] for word in words]

            keys_tensor = tf.constant(words, dtype=tf.int32)
            values_tensor = tf.constant(vectors, dtype=tf.float32)
            zero_vector = tf.zeros_like(values_tensor[0], dtype=tf.float32)

            print("keys_tensor", keys_tensor.shape, "values_tensor", values_tensor.shape, "zero_vector", zero_vector.shape)

            lookup_table = tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int32,
                value_dtype=tf.float32,
                default_value=zero_vector,
                empty_key=-1,
                deleted_key=-2
            )
            lookup_table.insert(keys_tensor, values_tensor)

            print("model_key", model_key)
            print(model_key.ref())
            lookup_tables[model_key.numpy()] = lookup_table

        print("lookup_tables", lookup_tables.keys())
        return lookup_tables

    @staticmethod
    def _get_angles(pos, i, dim_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim_model))
        return pos * angle_rates

    @staticmethod
    def _positional_encoding(sentence_length, dim_model):
            angle_rads = TransformerWord2VecEncoder._get_angles(np.arange(sentence_length)[:, np.newaxis],
                                        np.arange(dim_model)[np.newaxis, :],
                                        dim_model)
            # apply sin to even indices in the array; 2i
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            # apply cos to odd indices in the array; 2i+1
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
            pos_encoding = angle_rads[np.newaxis, ...]
            
            return tf.cast(pos_encoding, dtype=tf.float32)

    def reshape_to_attributes_first(self, input_tensor):
        batch_size = tf.shape(input_tensor)[0]
        reshaped = tf.reshape(
            input_tensor, 
            (batch_size, self.case_length, self.num_attributes)
        )
        transposed = tf.transpose(reshaped, perm=[2, 1, 0])  # `[num_attributes, case_length, batch_size]`
        final_shape = tf.reshape(transposed, (self.num_attributes, self.case_length * batch_size))
        return final_shape

    def expand_to_dim_model(self, input_tensor):
        unique_masks = tf.unique(self.attribute_mask).y  # Extract unique mask values
        expanded_tensors = []  # To store expanded tensors for each group

        for mask_value in unique_masks:
            # Create a mask for the current attribute value
            group_mask = self.attribute_mask == mask_value
            group_tensor = tf.boolean_mask(input_tensor, group_mask, axis=0)

            if mask_value == -1:
                # Expand numerical values to `dim_model`
                expanded_group_tensor = tf.expand_dims(group_tensor, axis=-1)
                expanded_group_tensor = tf.tile(expanded_group_tensor, [1, 1, self.dim_model])
                # print("expanded_group_tensor_numerical", expanded_group_tensor.shape)
            else:
                # Expand categorical values to `dim_model`
                group_tensor = tf.cast(group_tensor, dtype=tf.int32)
                group_tensor = tf.reshape(group_tensor, [-1]) # Flatten
                # print("group_tensor", group_tensor.shape)
                # print(group_tensor)
                expanded_group_tensor = self.lookup_tables[mask_value.numpy()].lookup(group_tensor)
                expanded_group_tensor = tf.expand_dims(expanded_group_tensor, axis=0)

                # print("expanded_group_tensor_categorical", expanded_group_tensor.shape)
            
            expanded_tensors.append(expanded_group_tensor)

        # Combine all groups back together
        result = tf.concat(expanded_tensors, axis=0)
        # print("expanded_tensors", result.shape)

        return result

    def reshape_to_batch_first(self, input_tensor, batch_size):
        reshaped = tf.reshape(
            input_tensor, 
            (self.num_attributes, self.case_length, batch_size, self.dim_model)
        )
        transposed = tf.transpose(reshaped, perm=[2, 1, 0, 3])  # `[batch_size, case_length, num_attributes, dim_model]`
        final_shape = tf.reshape(
            transposed, 
            (batch_size, self.case_length * self.num_attributes, self.dim_model)
        )
        return final_shape

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Step 1: Reshape to `[num_attributes, case_length * batch_size]`
        attributes_first = self.reshape_to_attributes_first(inputs)

        # Step 2: Expand to `[num_attributes, case_length * batch_size, dim_model]`
        expanded_tensor = self.expand_to_dim_model(attributes_first)

        # Step 3: Transform back to `[batch_size, case_length * num_attributes, dim_model]`
        output_tensor = self.reshape_to_batch_first(expanded_tensor, batch_size)

        return output_tensor + self.positional_matrix_trace