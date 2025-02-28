from tensorflow import math, cast, float32, newaxis
from keras import Model
from keras.layers import Dense

from novel.transformer.components.encoder import Encoder
from novel.transformer.components.positional_w2v_encoding import TransformerWord2VecEncoder
from utils.enums import AttributeType

# https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/

class TransformerModel(Model):
    def __init__(self, encoders, attribute_types, attribute_vocab_sizes, attribute_keys_input, attribute_keys_output, event_positional_encoding, enc_seq_length, num_heads, dim_queries_keys, dim_values, dim_model, dim_feed_forward, num_layers, dropout_rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        # Set up the input encoding
        self.input_encoding = TransformerWord2VecEncoder(
            attribute_keys=attribute_keys_input,
            sequence_length=enc_seq_length,
            encoders=encoders,
            dim_model=dim_model,
            event_positional_encoding=event_positional_encoding,
        )

        # Set up the encoder
        self.encoder = Encoder(
            sequence_length=enc_seq_length,
            num_heads=num_heads, 
            dim_queries_keys=dim_queries_keys, 
            dim_values=dim_values, 
            dim_model=dim_model, 
            dim_feed_forward=dim_feed_forward, 
            num_layers=num_layers, 
            dropout_rate=dropout_rate
        )

        # Set up the multi-task layers
        # Every attribute should be a seperate task
        # print("Multi-Task Outputs:")
        self.multi_task_outputs = []
        for attribute_type, attribute_key, attribute_vocab_size in zip(attribute_types, attribute_keys_output, attribute_vocab_sizes):
            print(attribute_type, attribute_key, attribute_vocab_size)
            if attribute_type == AttributeType.NUMERICAL:
                # If numerical: Output a single value
                # Use sigmoid for numerical outputs as it scales the output between 0 and 1
                self.multi_task_outputs.append(
                    Dense(attribute_vocab_size, activation='sigmoid', name=attribute_key)
                ) 
            elif attribute_type == AttributeType.CATEGORICAL:
                # If categorical: Output a softmax layer with the size of the vocabulary
                # Use softmax for categorical outputs as it scales the output between 0 and 1 and ensure all classes sum up to 1
                self.multi_task_outputs.append(
                    Dense(attribute_vocab_size, activation='softmax', name=attribute_key)
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

    def call(self, input_trace, training): # decoder_input
        # training is a boolean flag that indicates if the model is in training mode
        # encoder_input and decoder_input are the inputs to the encoder and decoder respectively

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(input_trace)

        # Generate the positional encoding
        # print("encoding_start", input_trace.shape)
        encoding_output = self.input_encoding(input_trace)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # print("encoding_output", encoding_output.shape)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoding_output, enc_padding_mask, training)

        # Pass the decoder output through the multi-task layers
        model_outputs = []
        for i, task in enumerate(self.multi_task_outputs):
            attribute_output = encoder_output[:, i] # decoder_output[:, i]
            model_outputs.append(task(attribute_output))

        return model_outputs