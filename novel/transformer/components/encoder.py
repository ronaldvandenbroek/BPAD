from keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout, Input
from keras import Model

from novel.transformer.components.attention import MultiHeadAttention
from novel.transformer.components.positional_encoding import PositionEmbeddingFixedWeights, PositionWordEmbeddingFixedWeights

# https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, dim_feed_forward, dim_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(dim_feed_forward)  # First fully connected layer
        self.fully_connected2 = Dense(dim_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, sequence_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, dropout_rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        # To print out summary:
        self.sequence_length =sequence_length
        self.d_model=dim_model
        self.build(input_shape=[None, sequence_length, dim_model])

        self.multihead_attention = MultiHeadAttention(num_heads, dim_queries_keys, dim_values, dim_model)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(dim_feed_forward, dim_model)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)
    
    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, sequence_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, enc_vocab_size=None, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        if enc_vocab_size is not None:
            self.pos_encoding = PositionWordEmbeddingFixedWeights(sequence_length, enc_vocab_size, dim_model)
        else:
            self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, dim_model)
        self.dropout = Dropout(dropout_rate)
        self.encoder_layer = [EncoderLayer(sequence_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, dropout_rate) for _ in range(num_layers)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x