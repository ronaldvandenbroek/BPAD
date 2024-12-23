import numpy as np
from baseline.binet.core import NNAnomalyDetector
from novel.transformer.components.transformer import TransformerModel
from novel.transformer.components.utils import LRScheduler
from utils import process_results
from utils.dataset import Dataset
from utils.embedding.attribute_dictionary import AttributeDictionary
from utils.enums import EncodingCategorical
from utils.settings.settings_multi_task import SettingsMultiTask
from utils.enums import AttributeType

from keras.optimizers import Adam # type: ignore
from keras.metrics import Mean # type: ignore
from tensorflow import data, train, GradientTape, function # type: ignore

from novel.transformer.components.transformer import TransformerModel
from novel.transformer.components.utils import LRScheduler, loss_fcn, accuracy_fcn, likelihood_fcn


class Transformer(NNAnomalyDetector):
    """Implements a transformer based anomaly detection algorithm."""

    abbreviation = 'transformer'
    name = 'Transformer'
    supports_attributes = True

    config = dict(        
        # Model Paramerters
        num_heads = 8,  # Number of self-attention heads
        dim_queries_keys = 32, #64,  # Dimensionality of the linearly projected queries and keys
        dim_values = 32, #64,  # Dimensionality of the linearly projected values
        dim_model = 256, #512,  # Dimensionality of model layers' outputs
        dim_feed_forward = 512, #2048,  # Dimensionality of the inner fully connected layer
        num_layers = 2, #3, #6,  # Number of layers in the encoder/decoder stack

        # Training Parameters
        dropout_rate = 0.1,
        batch_size = 8,
        beta_1 = 0.9,
        beta_2 = 0.98,
        epsilon = 1e-7, #1e-9,
     ) 
    
    def __init__(self, model=None):
        """Initialize Trasformer model.

        Size of hidden layers is based on input size. The size can be controlled via the hidden_size_factor parameter.
        This can be float or a list of floats (where len(hidden_size_factor) == hidden_layers). The input layer size is
        multiplied by the respective factor to get the hidden layer size.

        :param model: Path to saved model file. Defaults to None.
        :param hidden_layers: Number of hidden layers. Defaults to 2.
        :param hidden_size_factor: Size factors for hidden layer base don input layer size.
        :param epochs: Number of epochs to train.
        :param batch_size: Mini batch size.
        """
        super(Transformer, self).__init__(model=model)

    @staticmethod
    def model_fn(dataset:Dataset, **kwargs):
        print(f"Config: {kwargs}")

        # Determine the vocab size
        vocab_size = 0
        for encoder in dataset.encoders.values():
            print(encoder)
            encoder:AttributeDictionary
            largest_attribute = encoder.largest_attribute()
            if largest_attribute > vocab_size:
                vocab_size = largest_attribute
        vocab_size += 1

        print(f"Vocab size: {vocab_size}")

        # Configuring the dataset
        features = np.array(dataset.features)
        cases = np.transpose(features, (1, 2, 0))

        # Create a boolean mask for categorical attributes
        # Categorical filter TODO make sure the transformer model can also handle numerical values via multitask learning
        categorical_mask = np.array(dataset.attribute_types) == AttributeType.CATEGORICAL
        print(f"Categorical Mask: {categorical_mask}")
        num_categorical_features = np.sum(categorical_mask)
        print(f"Num Categorical Features: {num_categorical_features}")
        # Use this mask to filter out only the categorical attributes in the last axis
        categorical_cases = cases[..., categorical_mask]

        attribute_perspectives = np.array(dataset.event_log.event_attribute_perspectives)[categorical_mask]
        attribute_dims = np.array([1] * len(attribute_perspectives))
        print("attribute_dims: ", attribute_dims)
        print("attribute_perspectives: ", attribute_perspectives)

        zero_event = np.zeros((num_categorical_features))
        trainX_categorical = []
        trainY_categorical = []
        for index, (case, case_length) in enumerate(zip(categorical_cases, dataset.case_lens)):
            last_event_index = case_length - 1
            trainY_categorical.append(case[last_event_index].copy())

            # Remove the target event from the training data
            case[last_event_index] = zero_event
            trainX_categorical.append(case)

        trainX_categorical = np.array(trainX_categorical, dtype=np.int64)
        trainY_categorical = np.array(trainY_categorical, dtype=np.int64)

        process_results.check_array_properties("trainX_categorical", trainX_categorical)
        process_results.check_array_properties("trainY_categorical", trainY_categorical)

        dim0, dim1, dim2 = trainX_categorical.shape
        trainX_categorical = np.reshape(trainX_categorical, (dim0, dim1 * dim2))#, order='C')

        enc_seq_length = trainX_categorical.shape[1]
        dec_seq_length = trainY_categorical.shape[1]
        print(trainX_categorical.shape, trainY_categorical.shape)
        
        train_dataset = data.Dataset.from_tensor_slices((trainX_categorical, trainY_categorical))
        train_dataset = train_dataset.batch(kwargs.get('batch_size'))
            
        # Configuring the model
        model = TransformerModel(
            # Dataset variables
            enc_vocab_size=vocab_size,
            dec_vocab_size=vocab_size,
            enc_seq_length=enc_seq_length,
            dec_seq_length=dec_seq_length,
            # Model variables
            num_heads=kwargs.get('num_heads'),
            dim_queries_keys=kwargs.get('dim_queries_keys'),
            dim_values=kwargs.get('dim_values'),
            dim_model=kwargs.get('dim_model'),
            dim_feed_forward=kwargs.get('dim_feed_forward'),
            num_layers=kwargs.get('num_layers'),
            dropout_rate=kwargs.get('dropout_rate'),
        )

        optimizer = Adam(
            LRScheduler(d_model=kwargs.get('dim_model')), 
            beta_1=kwargs.get('beta_1'), 
            beta_2=kwargs.get('beta_2'), 
            epsilon=kwargs.get('epsilon')
        )

        # Create a checkpoint object and manager to manage multiple checkpoints
        # RCVDB TODO: Check if this is useful in an online setting
        # ckpt = train.Checkpoint(model=model, optimizer=optimizer)
        # ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

        return model, optimizer, train_dataset, num_categorical_features, attribute_dims, attribute_perspectives # , trainY_categorical
    
    def _train_and_predict(self, model, optimizer, train_dataset, num_categorical_features):
        train_loss = Mean(name='train_loss')
        # train_accuracy = Mean(name='train_accuracy')
        train_likelihood = Mean(name='train_likelihood')

        predictions = []
        targets = []
        losses = []

        # Speeding up the training process
        @function
        def train_step(encoder_input, decoder_input, decoder_output):
            with GradientTape() as tape:
                prediction = model(encoder_input, decoder_input, training=True)

                loss = loss_fcn(decoder_output, prediction)
                # accuracy = accuracy_fcn(decoder_output, prediction)
                likelihood = likelihood_fcn(decoder_output, prediction)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            train_loss(loss)
            # train_accuracy(accuracy)
            train_likelihood(likelihood)

            return loss, prediction

        train_loss.reset_states()
        # train_accuracy.reset_states()
        train_likelihood.reset_states()

        print(f"\nStart of training in {len(train_dataset)} batches")

        for step, (train_batchX, train_batchY) in enumerate(train_dataset):
            encoder_input = train_batchX[:, num_categorical_features:] #1:] #Skip the start symbol
            decoder_input =  train_batchY[:, 0:num_categorical_features]    #train_batchY[:, :-1] 
            decoder_output = train_batchY[:, :]

            # print(encoder_input.shape, "Encoder Input")
            # print(decoder_input.shape, "Decoder Input")
            # print(decoder_output.shape, "Decoder Output")

            loss, prediction = train_step(encoder_input, decoder_input, decoder_output)

            losses.append(loss.numpy())
            predictions.append(prediction.numpy())
            targets.append(decoder_output.numpy())

            if step % 25 == 0:
                print(f'Step {step} Loss {train_loss.result():.4f} Likelihood {train_likelihood.result():.4f}') # Accuracy {train_accuracy.result():.4f}

        # Print epoch number and loss value at the end of every epoch
        print("Training Loss %.4f, Training Likelihood %.4f" % (train_loss.result(), train_likelihood.result())) # train_accuracy.result(),

        return np.vstack(predictions), np.vstack(losses), np.vstack(targets),

    def train_and_predict(self, 
                          dataset:Dataset, 
                          batch_size=None, 
                          bucket_boundaries=None,
                          categorical_encoding=EncodingCategorical.TOKENIZER, 
                          vector_size=None):
        
        if batch_size != None:
            self.config['batch_size'] = batch_size
        
        model, optimizer, train_dataset, num_categorical_features, attribute_dims, attribute_perspectives = self.model_fn(dataset, **self.config)


        case_lengths = dataset.case_lens
        # case_max_length = dataset.max_len
        attribute_names = dataset.event_log.event_attribute_keys

        predictions, losses, targets = self._train_and_predict(model, optimizer, train_dataset, num_categorical_features)

        print(predictions.shape, targets.shape)
        print(targets)

        likelihood_errors = np.zeros(targets.shape)
        for i, (event_pred, event_true) in enumerate(zip(predictions, targets)):
            for j, (attribute_pred, attribute_true) in enumerate(zip(event_pred, event_true)):
                likelihood_true = attribute_pred[attribute_true]
                likelihood_pred = 0
                for pred in attribute_pred:
                    if pred > likelihood_true:
                        likelihood_pred += pred
                likelihood_errors[i,j] = likelihood_pred

        print(likelihood_errors.shape)   
        # likelihood_errors = np.transpose(likelihood_errors, (1, 0))
        perspective_likelihood_errors = likelihood_errors.copy()
        print(perspective_likelihood_errors.shape)

        # Check predictions, losses, targets, likelihood_errors
        process_results.check_array_properties("Predictions", predictions)
        process_results.check_array_properties("Losses", losses)
        process_results.check_array_properties("Targets", targets)
        process_results.check_array_properties("Likelihood Errors", likelihood_errors)

        trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = process_results.process_bucket_results(
                errors_raw=perspective_likelihood_errors,
                categorical_encoding=categorical_encoding,
                vector_size=None, # As no vector size is used for next event prediction
                bucketing=False, # As no bucketing is done for the transformer model
                attribute_dims=attribute_dims,
                dataset_mask=None, # dataset.mask, # As only one new event is predicted
                attribute_types=dataset.attribute_types,
                case_max_length=1, # As only one new event is predicted
                anomaly_perspectives=attribute_perspectives,
                case_lengths=case_lengths,
                error_power=2
            )
        
        case_labels = dataset.case_labels
        event_labels = dataset.event_labels
        attr_labels = dataset.attr_labels
        
        return ({0:trace_level_abnormal_scores}, 
                {0:event_level_abnormal_scores}, 
                {0:attr_level_abnormal_scores}, 
                {0:losses},
                {0:case_labels}, 
                {0:event_labels}, 
                {0:attr_labels}, 
                {0:perspective_likelihood_errors},
                attribute_perspectives,
                attribute_perspectives,
                attribute_names,
                attribute_names)
    
