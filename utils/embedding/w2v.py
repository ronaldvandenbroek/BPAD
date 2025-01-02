from gensim.models import Word2Vec
import numpy as np

from sklearn.decomposition import PCA
import tensorflow as tf
from tqdm import tqdm

from processmining.case import Case
from processmining.log import EventLog
from utils.embedding.attribute_dictionary import AttributeDictionary
from utils.embedding.util import fourier_encoding, recalculate_attribute_dimensions
from utils.enums import AttributeType
from utils.fs import FSSave

from keras.layers import Layer

class TransformerWord2VecEncoder(Layer):
    def __init__(self, attribute_keys, sequence_length, encoders, dim_model=50, **kwargs) -> None:
        super(TransformerWord2VecEncoder, self).__init__(**kwargs)

        self.dim_model = dim_model
        self.sequence_length = sequence_length

        self.num_attributes = len(attribute_keys)
        case_length = sequence_length / self.num_attributes
        self.case_length = tf.cast(case_length, tf.int32)
        print("case_length", self.case_length)

        # Create a dictionary to map attribute keys to their index
        attribute_key_dict, attribute_keys_tensor = TransformerWord2VecEncoder._convert_encoder_keys(encoders)

        self.attribute_key_trace_mask, self.categorical_mask, self.numerical_mask, self.attribute_mask = TransformerWord2VecEncoder._create_attribute_masks(
            attribute_keys, attribute_key_dict, self.case_length)

        extracted_w2v_models = TransformerWord2VecEncoder._create_w2v_models(encoders, dim_model)
        self.lookup_tables = TransformerWord2VecEncoder._create_lookup_tables(attribute_keys_tensor, extracted_w2v_models)

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

        return output_tensor

    
class ProcessWord2VecEncoder():
    def __init__(self,
                 encoders, 
                 attribute_types, 
                 event_attribute_keys,
                 features,
                 event_log:EventLog,
                 pretrain_percentage=0.001,
                 vector_size=50, 
                 window=5, 
                 min_count=1, 
                 workers=4,
                 fs_save:FSSave=None) -> None:
        #self.encoders = encoders
        self.attribute_types = attribute_types
        self.event_attribute_keys = event_attribute_keys
        self.features = features

        # self.event_log = event_log
        self.pretrain_percentage = pretrain_percentage

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

        self.fs_save=fs_save

        # Precompute
        self.frequencies = np.linspace(1, vector_size / 2, vector_size // 2)
        self.zero_vector = np.zeros(vector_size, dtype=np.float32)

        # Setup models
        self.w2v_models = {}
        percentage_index = int(np.ceil((pretrain_percentage * len(event_log.cases))))
        pretrain_cases = event_log.cases[:percentage_index]
        for attribute_type, attribute_key in zip(self.attribute_types, self.event_attribute_keys):
            if attribute_type == AttributeType.CATEGORICAL:
                pretrain_sentences = self.create_training_data(
                    pretrain_cases=pretrain_cases,
                    attribute_key=attribute_key,
                    encoder=encoders[attribute_key])

                w2v_model:Word2Vec = self.create_model(pretrain_sentences)
                self.w2v_models[attribute_key] = w2v_model

                # if fs_save is not None:
                #     self._save_embedding_space(w2v_model, attribute_key, pretrain_percentage)

    def create_training_data(self, pretrain_cases, attribute_key, encoder:AttributeDictionary):
        pretrain_sentences = []
        for pretrain_case in pretrain_cases:
            pretrain_case:Case
            pretrain_sentence = pretrain_case.get_attributes_by_type(attribute_key)
            pretrain_sentence_encoded = encoder.encode_list(pretrain_sentence)
            pretrain_sentences.append(pretrain_sentence_encoded)
        # Append the unused buffer values
        # return self._convert_to_sentences(encoder.encoded_attributes() + encoder.buffer_attributes())
        return pretrain_sentences + self._convert_to_sentences(encoder.encoded_attributes() + encoder.buffer_attributes())

    def _convert_to_sentences(self, input):
        return [[str(i)] for i in input]

    def create_model(self, training_sentences):
        return Word2Vec(
            sentences=training_sentences, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=self.workers,
            sg=1, # Use Skipgram
            hs=0, # Heirarchical Softmax
            negative=0
            )

    def encode_attribute(self, input, attribute_key):
        model:Word2Vec = self.w2v_models[attribute_key]
        try:
            return model.wv[str(int(input))]
        except KeyError as e:
            print(attribute_key, input)
            print(model.wv.index_to_key)
            raise e
    
    def encode_features(self, average=True, match_numerical=False):
        w2v_features = []
        w2v_feature_names = []
        numeric_features = []
        numeric_feature_names = []
        for index, (feature, attribute_type, attribute_key) in tqdm(enumerate(zip(self.features, self.attribute_types, self.event_attribute_keys)), "Encoding W2V Features"):
            # RCVDB: TODO Experimental, not encoding start and end event
            # feature_experimental = []
            # for case in feature:
            #     feature_experimental.append(case[1:-1])
            # feature = feature_experimental  

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
                        if attr != 0: # Attributes with 0 are from events that do not exist
                            # print(attribute_key)
                            w2v_attr_vector = self.encode_attribute(attr, attribute_key)
                            trace_attributes.append(np.array(w2v_attr_vector, dtype=np.float32))
                        elif average == False:
                            trace_attributes.append(self.zero_vector)

                    if average:
                        # Average the attribute value over all events
                        encoded_feature.append(np.mean(np.vstack(trace_attributes), axis=0))
                    else:
                        encoded_feature.append(trace_attributes)
                w2v_features.append(np.array(encoded_feature, dtype=np.float32))
                w2v_feature_names.append(attribute_key) 
            
        return np.array(w2v_features, dtype=np.float32), np.array(numeric_features, dtype=np.float32), np.array(numeric_feature_names), np.array(w2v_feature_names)

    def encode_flat_features_2d(self, attribute_keys, trace2vec=False, match_numerical=False):
        # if numerical_encoding == Vector: w2v_features, numeric_features, numeric_feature_names, w2v_feature_names = self.encode_features(average=False, match_numerical=True)
        w2v_features, numeric_features, numeric_feature_names, w2v_feature_names = self.encode_features(average=False, match_numerical=match_numerical)
        # print(w2v_features.shape, numeric_features.shape)

        # print(transposed_w2v_features.shape, transposed_numeric_features.shape)
        # print(numeric_features.shape, numeric_feature_names)
        # print(w2v_features.shape, w2v_feature_names)
        # print(attribute_keys)

        transposed_w2v_features = np.transpose(w2v_features, (1, 2, 0, 3))
        transposed_numeric_features = np.transpose(numeric_features, (1, 2, 0))

        # print(transposed_numeric_features.shape, numeric_feature_names)
        # print(transposed_w2v_features.shape, w2v_feature_names)
        # print(attribute_keys)
        
        reordered_slices = []
        numeric_feature_names = list(numeric_feature_names)
        w2v_feature_names = list(w2v_feature_names)
        # Iterate through each feature in the total_order
        for feature in attribute_keys:
            if feature in numeric_feature_names:
                feature_index = numeric_feature_names.index(feature)
                if match_numerical:
                    reordered_slices.append(transposed_w2v_features[:, :, feature_index, :])  # Shape (35483, 13, 200)
                else:
                    reordered_slices.append(transposed_numeric_features[:, :, feature_index:feature_index+1])  # Retain shape (35483, 13, 1)
            elif feature in w2v_feature_names:
                feature_index = w2v_feature_names.index(feature)
                reordered_slices.append(transposed_w2v_features[:, :, feature_index, :])  # Shape (35483, 13, 200)

        # Concatenate all slices along the last axis
        merged_features = np.concatenate(reordered_slices, axis=-1)

        # Generates a single encoding per case that can be prepended to the w2v encoding
        trace_encoding = np.mean(merged_features, axis=(1))

        print(merged_features.shape)
        dim0, dim1, dim2 = merged_features.shape
        flat_merged_features = np.reshape(merged_features, (dim0, dim1 * dim2))#, order='C')
        print(flat_merged_features.shape)

        if trace2vec:
            # (num_cases, trace_encoding + num_attribute * num_events)
            flat_merged_features = np.concatenate((trace_encoding,flat_merged_features), axis=1)  

        print(flat_merged_features.shape)

        return flat_merged_features, recalculate_attribute_dimensions(self.attribute_types, self.vector_size, sort=False, match_numerical=match_numerical)


    def encode_flat_features_2d_average(self, trace2vec=False):
        w2v_features, numeric_features, numeric_feature_names, w2v_feature_names = self.encode_features()

        # RCVDB: Interleaf the w2v features
        # (num_attribute, num_cases, vector_size) 
        # (num_cases, num_attribute, vector_size) 
        # (num_cases, num_attribute * vector_size) 
        transposed_w2v_features = np.transpose(w2v_features, (1, 0, 2))
        dim0, dim1, dim2 = transposed_w2v_features.shape
        # Need order C so the whole w2v of each attribute is next to each other
        flat_w2v_features = np.reshape(transposed_w2v_features, (dim0, dim1 * dim2), order='C')

        # RCVDB: Interleaf the numeric features
        # (num_attribute, num_cases, num_events) 
        # (num_cases, num_attribute, num_events) 
        # (num_cases, num_attribute * num_events)
        if len(numeric_features) > 0:
            transposed_numeric_features = np.transpose(numeric_features, (1, 0, 2))
            dim0, dim1, dim2 = transposed_numeric_features.shape
            # Need order F so each event but not each attribute of the same type are next to eachother
            flat_numeric_features = np.reshape(transposed_numeric_features, (dim0, dim1 * dim2), order='F')

            flat_w2v_numeric_features = np.concatenate((flat_w2v_features,flat_numeric_features), axis=1)
        else:
            flat_w2v_numeric_features = flat_w2v_features

        if trace2vec:
            # Generates a single encoding per case that can be prepended to the w2v encoding
            # TODO: Error mean of empty slice in the smallest bucket
            trace_encoding = np.mean(transposed_w2v_features, axis=1)
            # (num_cases, trace_encoding + num_attribute * num_events)
            flat_w2v_numeric_features = np.concatenate((trace_encoding,flat_w2v_numeric_features), axis=1)      

        # RCVDB: Sanity check to see if all values are encoded correctly.
        assert not np.any(np.isnan(flat_w2v_numeric_features)), "Data contains NaNs!"
        assert not np.any(np.isinf(flat_w2v_numeric_features)), "Data contains Infs!"

        return flat_w2v_numeric_features, recalculate_attribute_dimensions(self.attribute_types, self.vector_size, sort=True, match_numerical=False)
    
    def _save_embedding_space(self, model, attribute_key, pretrain_percentage):
        words = list(model.wv.index_to_key)
        word_vectors = np.array([model.wv[word] for word in words])

        pca = PCA(n_components=2)
        word_vectors_2d = pca.fit_transform(word_vectors)

        self.fs_save.save_embedding_space(f"W2V_{attribute_key}", pretrain_percentage, words, word_vectors_2d)
        