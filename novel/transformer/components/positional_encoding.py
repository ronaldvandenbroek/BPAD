

import numpy as np
import tensorflow as tf
from keras.layers import Layer, Embedding

# https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/

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


