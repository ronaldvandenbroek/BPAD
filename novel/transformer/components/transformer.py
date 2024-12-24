from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from keras import Model
from keras.layers import Dense, Activation, Flatten

from novel.transformer.components.decoder import Decoder
from novel.transformer.components.encoder import Encoder
from utils.enums import AttributeType

# https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/

class TransformerModel(Model):
    def __init__(self, attribute_type_mask, attribute_types, attribute_keys, enc_seq_length, dec_seq_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, enc_vocab_size=None, dec_vocab_size=None, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        print(enc_vocab_size, "Enc Vocab Size")
        print(dec_vocab_size, "Dec Vocab Size")
        print(num_heads, "Num Heads")
        print(dim_queries_keys, "Dim Queries Keys")
        print(dim_values, "Dim Values")
        print(dim_model, "Dim Model")
        print(dim_feed_forward, "Dim Feed Forward")
        print(num_layers, "Num Layers")
        print(dropout_rate, "Dropout Rate")
        print(enc_seq_length, "Enc Seq Length")
        print(dec_seq_length, "Dec Seq Length")
        # RCVDB TODO: enc_vocab_size and dec_vocab_size should be an array for each categorical attribute when doing multi-task learning
        # PositionEmbeddingFixedWeights can probably be defined here and then passed through to the encoders and decoders

        # Set up the encoder
        self.encoder = Encoder(
            attribute_type_mask=attribute_type_mask,
            sequence_length=enc_seq_length,
            num_heads=num_heads, 
            dim_queries_keys=dim_queries_keys, 
            dim_values=dim_values, 
            dim_model=dim_model, 
            dim_feed_forward=dim_feed_forward, 
            num_layers=num_layers, 
            dropout_rate=dropout_rate, 
            enc_vocab_size=enc_vocab_size)

        # Set up the decoder
        # self.decoder = Decoder(dec_seq_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, dec_vocab_size=dec_vocab_size)

        # Set up the multi-task layers
        # Every attribute should be a seperate task
        # Init Input: Array of if inputs are numerical or categorical
        # For each attribute:
            # If numerical:
                # Output a single value
            # If categorical:
                # Output a softmax layer with the size of the vocabulary
        self.multi_task_outputs = []
        for attribute_type, attribute_key in zip(attribute_types, attribute_keys):
            if attribute_type == AttributeType.NUMERICAL:
                self.multi_task_outputs.append(Dense(1, activation='linear', name=attribute_key))
            elif attribute_type == AttributeType.CATEGORICAL:
                if dec_vocab_size is not None:
                    self.multi_task_outputs.append(
                        Dense(dec_vocab_size, activation='softmax', name=attribute_key)
                    )
                else:
                    self.multi_task_outputs.append(
                        Dense(dec_seq_length, activation='softmax', name=attribute_key)
                    )

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
        # training is a boolean flag that indicates if the model is in training mode
        # encoder_input and decoder_input are the inputs to the encoder and decoder respectively

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        # dec_in_padding_mask = self.padding_mask(decoder_input)
        # dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        # dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # RCVDV TODO: Add the positional encoding layer definitions here as each task will have a different encoding

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        # decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

        # Pass the decoder output through the multi-task layers
        model_outputs = []
        for i, task in enumerate(self.multi_task_outputs):
            attribute_output = encoder_output[:, i] # decoder_output[:, i]
            model_outputs.append(task(attribute_output))

        return model_outputs