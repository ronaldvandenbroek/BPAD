import numpy as np
import tensorflow as tf
from tqdm import tqdm
from baseline.binet.core import NNAnomalyDetector
from novel.transformer.components.prefix_store import PrefixStore
from novel.transformer.components.transformer import TransformerModel
from utils import process_results
from utils.dataset import Dataset

from keras.optimizers import Adam # type: ignore
from keras.metrics import Mean # type: ignore
from tensorflow import data, train, GradientTape, function # type: ignore
from keras.losses import sparse_categorical_crossentropy, log_cosh

from novel.transformer.components.transformer import TransformerModel


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
        learning_rate = 0.001, # 0.0001,
        dropout_rate = 0.1,
        batch_size = 8,
        beta_1 = 0.9,
        beta_2 = 0.98,
        epsilon = 1e-7, #1e-9,
     ) 
    
    def __init__(self, config):
        super(Transformer, self).__init__(config)

        # Update the config with the additional parameters
        self.config.update(config)

    def model_fn(self, dataset:Dataset):
        attribute_vocab_sizes = []
        attribute_keys = dataset.event_log.event_attribute_keys
        for attribute_key in attribute_keys:
            if attribute_key in dataset.encoders:
                attribute_vocab_sizes.append(dataset.encoders[attribute_key].max_size + 1)
            else:
                attribute_vocab_sizes.append(1)

        print(f"Attribute vocab sizes: {attribute_vocab_sizes}")

        # Configuring the dataset
        features = np.array(dataset.features)
        cases = np.transpose(features, (1, 2, 0))
        
        # Replace all zeros with a small value to avoid triggering the padding mask
        cases = np.where(cases == 0, 1e-7, cases)

        case_length = cases.shape[1]
        num_features = cases.shape[2]
        attribute_types = dataset.attribute_types
        attribute_perspectives = dataset.event_log.event_attribute_perspectives
        attribute_dims = np.array([1] * len(attribute_perspectives))

        enc_seq_length = (case_length - 1) * num_features
        dec_seq_length = num_features
        print(case_length, num_features)
        print(enc_seq_length, dec_seq_length)

        zero_event = np.zeros((num_features), dtype=np.float64)
        trainX = []
        trainY = []
        trainEventIndices = []
        for index, (case, case_length) in enumerate(zip(cases, dataset.case_lens)):
            last_event_index = case_length - 1
            trainY.append(case[last_event_index].copy())
            trainEventIndices.append(last_event_index * num_features)

            # Remove the target event from the training data and set the padding for all future events
            case[last_event_index:] = zero_event
            trainX.append(case)

        trainX = np.array(trainX, dtype=np.float64)
        trainY = np.array(trainY, dtype=np.float64)
        trainEventIndices = np.array(trainEventIndices, dtype=np.int32)

        process_results.check_array_properties("trainX_categorical", trainX)
        process_results.check_array_properties("trainY_categorical", trainY)

        dim0, dim1, dim2 = trainX.shape
        trainX = np.reshape(trainX, (dim0, dim1 * dim2))#, order='C')

        case_lengths = dataset.case_lens
        case_labels = dataset.case_labels
        event_labels = dataset.event_labels
        attr_labels = dataset.attr_labels

        # Capping the training data for development iteration purposes, if None then the full dataset is used
        limit = self.config.get('case_limit', None)
        if limit is not None:
            trainX = trainX[:limit]
            trainY = trainY[:limit]
            case_lengths = case_lengths[:limit]
            case_labels = case_labels[:limit]
            event_labels = event_labels[:limit]
            attr_labels = attr_labels[:limit]
            print("train_shapes after capping", trainX.shape, trainY.shape)

        train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
        train_dataset = train_dataset.batch(self.config.get('batch_size'))

        # RCVDB TODO: If single task model then attribute type and keys should be looped and the model only passed a single attribute type and key
        # Then a list of models per attribute should be created
    
        # Configuring the model
        model = TransformerModel(
            # Dataset variables
            encoders=dataset.encoders,
            # attribute_type_mask=attribute_type_mask,
            attribute_types=attribute_types,
            attribute_keys=attribute_keys,
            attribute_vocab_sizes=attribute_vocab_sizes,
            enc_seq_length=enc_seq_length,
            event_positional_encoding=self.config.get('event_positional_encoding'),
            # Model variables
            num_heads=self.config.get('num_heads'),
            dim_queries_keys=self.config.get('dim_queries_keys'),
            dim_values=self.config.get('dim_values'),
            dim_model=self.config.get('dim_model'),
            dim_feed_forward=self.config.get('dim_feed_forward'),
            num_layers=self.config.get('num_layers'),
            dropout_rate=self.config.get('dropout_rate'),
        )

        optimizer = Adam(
            learning_rate=self.config.get('learning_rate'), 
            beta_1=self.config.get('beta_1'), 
            beta_2=self.config.get('beta_2'), 
            epsilon=self.config.get('epsilon')
        )

        if self.config.get('debug_logging', False):
            print("Model Config", self.config)
            print(attribute_vocab_sizes, "Vocab Sizes")
            print(enc_seq_length, "Enc Seq Length")

        return model, optimizer, train_dataset, num_features, attribute_dims, attribute_perspectives, trainX, trainY, case_lengths, case_labels, event_labels, attr_labels
    
    def _train_and_predict(self, model, optimizer, train_dataset, num_features):
        print(num_features, "Num Features")

        train_loss = Mean(name='train_loss')
        train_loss.reset_states()

        predictions = {key: [] for key in range(num_features)}
        targets = []
        losses = []

        # Speeding up the training process
        # @function
        def train_step(encoder_input, decoder_output):
            with GradientTape() as tape:
                prediction_train_step = model(encoder_input, training=True)

                # print(len(prediction_train_step), prediction_train_step[0].shape, decoder_output.shape)

                loss_attributes = []
                for i, _ in enumerate(prediction_train_step):
                    pred = prediction_train_step[i]
                    true = decoder_output[:, i]

                    # print(pred.shape, true.shape, pred.shape[1])

                    if pred.shape[1] == 1:
                        loss_attribute = log_cosh(true, pred)
                    else:
                        # from_logits = False as the output layer of the model is softmax 
                        loss_attribute = sparse_categorical_crossentropy(true, pred, from_logits=False)
                    # print(loss_attribute.shape)

                    # RCVDB TODO: Could weight the different attribute by perspective
                    loss_attributes.append(tf.reduce_mean(loss_attribute))

                loss_train_step = tf.reduce_mean(loss_attributes)

            gradients = tape.gradient(loss_train_step, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            train_loss(loss_train_step)

            return loss_train_step, prediction_train_step

        print(f"\nStart of training in {len(train_dataset)} batches")
        for step, (train_batchX, train_batchY) in enumerate(train_dataset):
            model_input = train_batchX[:, num_features:] # Skip the starting event as it is always the same
            model_target = train_batchY[:, :]

            # print("Model Input Shape: " , model_input.shape, " Model Output Shape: ", model_target.shape)

            loss_train_step, predictions_train_step = train_step(model_input, model_target)

            for i, _ in enumerate(predictions_train_step):
                for j, _ in enumerate(predictions_train_step[i]):
                    predictions[i].append(predictions_train_step[i][j].numpy())

            losses.append(loss_train_step.numpy())
            targets.append(model_target.numpy())

            if step % 25 == 0:
                print(f'Step {step} Loss {train_loss.result():.4f}')

        # vstack the results, predictions are per attribute
        for prediction_key in predictions.keys():
            predictions[prediction_key] = np.vstack(predictions[prediction_key])
        return predictions, np.vstack(losses), np.vstack(targets)

    def train_and_predict(self, dataset:Dataset):
        debug_logging = self.config.get('debug_logging', False)

        model, optimizer, train_dataset, num_features, attribute_dims, attribute_perspectives, trainX, trainY, case_lengths, case_labels, event_labels, attr_labels = self.model_fn(dataset)
        predictions, losses, targets = self._train_and_predict(model, optimizer, train_dataset, num_features)

        # (traces, attributes, predictions)
        all_attribute_true = targets.T
        # (attributes, trace, predictions)

        # Convert categorical predictions into likelihoods
        # Convert numerical predictions into errors
        total_iterations = targets.shape[0] * targets.shape[1]
        attribute_errors = np.zeros(targets.shape)
        with tqdm(total=total_iterations, desc="Calculating Errors") as pbar:
            for i, (attribute_key, attribute_true) in enumerate(zip(predictions.keys(), all_attribute_true)):
                attribute_pred = predictions[attribute_key]
                for j, (trace_pred, trace_true) in enumerate(zip(attribute_pred, attribute_true)):
                    if trace_pred.shape[0] == 1: # If the prediction is a numerical value
                        pred = trace_pred[0]
                        true = trace_true
                        attribute_errors[j,i] = (pred - true)
                    else: # If the prediction is a categorical value
                        true = int(trace_true)
                        likelihood_true = trace_pred[true]
                        likelihood_pred = trace_pred[trace_pred > likelihood_true].sum()
                        attribute_errors[j,i] = likelihood_pred

                    pbar.update(1)

        print(attribute_errors.shape)
        print(trainX.shape, trainY.shape, len(case_lengths))

        # Convert the next event prediction errors into prefix errors
        prefix_store = PrefixStore(num_attributes=num_features, case_start_length=1, use_prefix_errors=self.config.get('use_prefix_errors'))
        prefix_store.add_prefixes(trainX, trainY, attribute_errors)
        attribute_errors = prefix_store.get_prefix_case_values()

        # Check if predictions, losses, targets, likelihood_errors are valid
        if debug_logging:
            process_results.check_array_properties("Losses", losses)
            process_results.check_array_properties("Targets", targets)
            process_results.check_array_properties("Errors", attribute_errors)

        trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = process_results.process_bucket_results(
                errors_raw=attribute_errors.copy(),
                categorical_encoding=self.config.get('categorical_encoding'),
                vector_size=None, # As no vector size is used for next event prediction
                bucketing=False, # As no bucketing is done for the transformer model
                attribute_dims=attribute_dims,
                dataset_mask=None, # dataset.mask, # As only one new event is predicted
                attribute_types=dataset.attribute_types,
                case_max_length=dataset.max_len, # As only one new event is predicted
                anomaly_perspectives=attribute_perspectives,
                case_lengths=case_lengths,
                error_power=2,
                debug_logging=debug_logging
            )
        
        attribute_names = dataset.event_log.event_attribute_keys
        results = ({0:trace_level_abnormal_scores}, 
                {0:event_level_abnormal_scores}, 
                {0:attr_level_abnormal_scores}, 
                {0:losses},
                {0:case_labels}, 
                {0:event_labels}, 
                {0:attr_labels}, 
                {0:attribute_errors},
                attribute_perspectives,
                attribute_perspectives,
                attribute_names,
                attribute_names,
                trainX,
                trainY)
        
        return results
    
