from keras.layers import Layer, Dropout, Input
from keras import Model

from novel.transformer.components.attention import MultiHeadAttention
from novel.transformer.components.positional_encoding import PositionEmbeddingFixedWeights, PositionWordEmbeddingFixedWeights
from novel.transformer.components.encoder import AddNormalization, FeedForward

# https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/

# Implementing the Decoder Layer
class DecoderLayer(Layer):
    def __init__(self, sequence_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, dropout_rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        # To print out summary:
        self.sequence_length =sequence_length
        self.d_model=dim_model
        self.build(input_shape=[None, sequence_length, dim_model])

        self.multihead_attention1 = MultiHeadAttention(num_heads, dim_queries_keys, dim_values, dim_model)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = AddNormalization()
        self.multihead_attention2 = MultiHeadAttention(num_heads, dim_queries_keys, dim_values, dim_model)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = AddNormalization()
        self.feed_forward = FeedForward(dim_feed_forward, dim_model)
        self.dropout3 = Dropout(dropout_rate)
        self.add_norm3 = AddNormalization()

    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):
        # Multi-head attention layer
        multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output1 = self.dropout1(multihead_output1, training=training)

        # Followed by an Add & Norm layer
        addnorm_output1 = self.add_norm1(x, multihead_output1)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by another multi-head attention layer
        multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output, padding_mask)

        # Add in another dropout layer
        multihead_output2 = self.dropout2(multihead_output2, training=training)

        # Followed by another Add & Norm layer
        addnorm_output2 = self.add_norm1(addnorm_output1, multihead_output2)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output2)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout3(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm3(addnorm_output2, feedforward_output)
    
    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, input_layer, None, None, True))

# Implementing the Decoder
class Decoder(Layer):
    def __init__(self, sequence_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, dec_vocab_size=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        if dec_vocab_size is not None:
            self.pos_encoding = PositionWordEmbeddingFixedWeights(sequence_length, dec_vocab_size, dim_model)
        else:
            self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, dim_model)
        self.dropout = Dropout(dropout_rate)
        self.decoder_layer = [DecoderLayer(sequence_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, dropout_rate) for _ in range(num_layers)]

    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(output_target)
        # Expected output shape = (number of sentences, sequence_length, d_model)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x