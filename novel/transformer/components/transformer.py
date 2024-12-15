from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from keras import Model
from keras.layers import Dense, Activation

from novel.transformer.components.decoder import Decoder
from novel.transformer.components.encoder import Encoder

# https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/

class TransformerModel(Model):
    def __init__(self, enc_seq_length, dec_seq_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, enc_vocab_size=None, dec_vocab_size=None, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_seq_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, enc_vocab_size=enc_vocab_size)

        # Set up the decoder
        self.decoder = Decoder(dec_seq_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, dec_vocab_size=dec_vocab_size)

        # Define the final dense layer
        if dec_vocab_size is not None:
            self.model_last_layer = Dense(dec_vocab_size)
        else:
            self.model_last_layer = Dense(dec_seq_length)
        # self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)
        mask = mask[:, newaxis, newaxis, :]
        # print(mask.shape, "Mask")

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        # Apply softmax to ensure probabilities for each token
        model_output = Activation('softmax')(model_output)  # Apply softmax along the last dimension

        return model_output