

import numpy as np
import tensorflow as tf
from keras.layers import Layer, Embedding

# https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/

class PositionMultiTaskEmbeddingLayer(Layer):
    def __init__(self, attribute_type_mask, sequence_length, vocab_size, dim_model, **kwargs):
        super(PositionMultiTaskEmbeddingLayer, self).__init__(**kwargs)
        self.categorical_mask = attribute_type_mask
        self.numerical_mask = tf.logical_not(attribute_type_mask)

        self.word_embedding_matrix = self.positional_encoding(vocab_size, dim_model)
        self.position_embedding_matrix = self.positional_encoding(sequence_length, dim_model)
        print(self.position_embedding_matrix.shape)

    def call(self, inputs):
        batch_size = inputs.shape[0]

        numerical_values = tf.where(self.numerical_mask, inputs, tf.zeros_like(inputs))
        categorical_values = tf.where(self.categorical_mask, inputs, tf.zeros_like(inputs))

        numerical_values = tf.expand_dims(numerical_values, -1)
        categorical_values = tf.expand_dims(categorical_values, -1)

        pos_encoding = tf.tile(self.position_embedding_matrix, [batch_size, 1, 1])

        numerical_pos_encoding = numerical_values + pos_encoding
        # RCVDB: TODO additionally encode categorical based on vocab sizes
        # categorical_word_encoding = categorical_values + 
        categorical_pos_encoding = categorical_values + pos_encoding

        combined_pos_encoding = tf.where(tf.expand_dims(self.categorical_mask, -1), categorical_pos_encoding, numerical_pos_encoding)

        # # Expand categorical_mask for broadcasting
        # expanded_categorical_mask = tf.expand_dims(self.categorical_mask, 0)  # Add batch dimension
        # expanded_categorical_mask = tf.expand_dims(expanded_categorical_mask, -1)  # Add model dimension
        # expanded_categorical_mask = tf.tile(
        #     expanded_categorical_mask, [tf.shape(inputs)[0], 1, tf.shape(self.position_embedding_matrix)[-1]]
        # )  # Shape: [batch_size, sequence_length, dim_model]

        # # Combine based on the mask
        # combined_pos_encoding = tf.where(
        #     expanded_categorical_mask,
        #     categorical_pos_encoding,
        #     numerical_pos_encoding
        # )

        print(combined_pos_encoding.shape, "Combined")

        # combined_pos_encoding = tf.where(self.categorical_mask, categorical_pos_encoding, numerical_pos_encoding)

        return combined_pos_encoding   

    def positional_encoding(self, sentence_length, dim_model):
        angle_rads = self.get_angles(np.arange(sentence_length)[:, np.newaxis],
                                     np.arange(dim_model)[np.newaxis, :],
                                     dim_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, dim_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim_model))
        return pos * angle_rates


class PositionWordEmbeddingLayer(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionWordEmbeddingLayer, self).__init__(**kwargs)
        # RCVDB: TODO Can use w2v here
        # RCVDB: TODO Probably need to implement the Attribute Dictionary here to make it as dynamic as possible
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices

# Attention is all you need version
class PositionWordEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, vocab_size, dim_model, **kwargs):
        super(PositionWordEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, dim_model)   
        position_embedding_matrix = self.get_position_encoding(sequence_length, dim_model)                                          
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=dim_model,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=dim_model,
            weights=[position_embedding_matrix],
            trainable=False
        )
             
    def get_position_encoding(self, sequence_length, dim_model, n=10000):
        P = np.zeros((sequence_length, dim_model))
        for k in range(sequence_length):
            for i in np.arange(int(dim_model/2)):
                denominator = np.power(n, 2*i/dim_model)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, inputs):
        print(inputs.shape, "Word Input")        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices

class PositionEmbeddingLayer(Layer):
    def __init__(self, sequence_length, dim_model, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)

        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=dim_model
        )

    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_indices = self.position_embedding_layer(position_indices)
        return inputs + embedded_indices
    
class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, dim_model, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)  # Correctly calls the parent class
        position_embedding_matrix = self.get_position_encoding(sequence_length, dim_model)
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, 
            output_dim=dim_model,
            weights=[position_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, sequence_length, dim_model, n=10000):
        P = np.zeros((sequence_length, dim_model))
        for k in range(sequence_length):
            for i in np.arange(int(dim_model / 2)):
                denominator = np.power(n, 2 * i / dim_model)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-2])
        embedded_words = inputs  # Ensure that `inputs` are the word embeddings passed in
        embedded_indices = self.position_embedding_layer(position_indices)
        print(inputs.shape, "Positional Input")
        print(embedded_indices.shape, "Positional Encoding")
        return embedded_words + embedded_indices


