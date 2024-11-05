# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from baseline.binet.core import NNAnomalyDetector
from utils.dataset import Dataset
from utils.enums import AttributeType, EncodingCategorical, Heuristic, Perspective, Strategy, Mode, Base
from collections import Counter, defaultdict

from utils.settings.settings_multi_task import SettingsMultiTask


class DAE(NNAnomalyDetector):
    """Implements a denoising autoencoder based anomaly detection algorithm."""

    abbreviation = 'dae'
    name = 'DAE'

    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE]
    supported_bases = [Base.LEGACY, Base.SCORES]
    supports_attributes = True

    config = dict(hidden_layers=2,
                  hidden_size_factor=.01,
                  noise=None # 0.5
     ) 

    def __init__(self, model=None):
        """Initialize DAE model.

        Size of hidden layers is based on input size. The size can be controlled via the hidden_size_factor parameter.
        This can be float or a list of floats (where len(hidden_size_factor) == hidden_layers). The input layer size is
        multiplied by the respective factor to get the hidden layer size.

        :param model: Path to saved model file. Defaults to None.
        :param hidden_layers: Number of hidden layers. Defaults to 2.
        :param hidden_size_factor: Size factors for hidden layer base don input layer size.
        :param epochs: Number of epochs to train.
        :param batch_size: Mini batch size.
        """
        super(DAE, self).__init__(model=model)

    @staticmethod
    def model_fn(dataset:Dataset, bucket_boundaries:list, categorical_encoding, vector_size=None, **kwargs):
        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')

        if categorical_encoding == EncodingCategorical.ONE_HOT:
            features = dataset.flat_onehot_features_2d
        elif categorical_encoding == EncodingCategorical.EMBEDDING:
            features = dataset.flat_embedding_features_2d
        elif categorical_encoding == EncodingCategorical.WORD_2_VEC_ATC:
            features = dataset.flat_w2v_features_2d_average()
        elif categorical_encoding == EncodingCategorical.WORD_2_VEC_C:
            features = dataset.flat_w2v_features_2d()
        elif categorical_encoding == EncodingCategorical.TRACE_2_VEC_ATC:
            features = dataset.flat_w2v_features_2d_average(trace2vec=True)
        elif categorical_encoding == EncodingCategorical.TRACE_2_VEC_C:
            features = dataset.flat_w2v_features_2d(trace2vec=True)
        elif categorical_encoding == EncodingCategorical.FIXED_VECTOR:
            features = dataset.flat_fixed_vector_features_2d()
        elif categorical_encoding == EncodingCategorical.ELMO:
            features = dataset.flat_elmo_features_2d()
        else:
            features = dataset.flat_features_2d

        case_lengths = dataset.case_lens
        case_labels = dataset.case_labels
        event_labels = dataset.event_labels
        attr_labels = dataset.attr_labels

        if bucket_boundaries is None:
            input_size = features.shape[1]
            # print(f"Input Size: {input_size}")

            model = DAE.build_model(
                input_size=input_size,
                hidden_layers=hidden_layers,
                hidden_size_factor=hidden_size_factor,
                noise=noise,
                multi_task=None)
            
            # Features are also targets in the case of reconstruction
            return {0:model}, {0:features}, {0:features}, {0:case_lengths}, {0:case_labels}, {0:event_labels}, {0:attr_labels}
        else:
            # Ensure that the longest length is also contained in the boundaries
            bucket_boundaries.append(dataset.max_len)
            print(f'Bucket Boundaries: {bucket_boundaries}')
            
            bucket_input_sizes = []
            bucket_ids = dataset.assign_to_buckets(bucket_boundaries)

            event_length = dataset.attribute_dims.sum()
            
            bucket_models = {}
            for i, boundary in enumerate(bucket_boundaries):
                if categorical_encoding == EncodingCategorical.WORD_2_VEC_ATC:
                    # If using w2v every categorical event length has the same shape as the events are aggregated
                    input_size = vector_size * dataset.attribute_type_count(AttributeType.CATEGORICAL) + boundary * dataset.attribute_type_count(AttributeType.NUMERICAL)
                elif categorical_encoding == EncodingCategorical.TRACE_2_VEC_ATC:
                    input_size = vector_size + vector_size * dataset.attribute_type_count(AttributeType.CATEGORICAL) + boundary * dataset.attribute_type_count(AttributeType.NUMERICAL)
                elif categorical_encoding == EncodingCategorical.TRACE_2_VEC_C:
                    input_size = vector_size + int(boundary * event_length)    
                else:
                    input_size = int(boundary * event_length)
                bucket_input_sizes.append(input_size)

                bucket_models[i] = DAE.build_model(
                                        input_size=input_size,
                                        hidden_layers=hidden_layers,
                                        hidden_size_factor=hidden_size_factor,
                                        noise=noise,
                                        multi_task=None)
            
            # Organize data into buckets and remove the unnessary padding
            bucket_features = {i: [] for i in range(len(bucket_boundaries))}
            bucket_case_lengths = {i: [] for i in range(len(bucket_boundaries))}
            bucket_case_labels = {i: [] for i in range(len(bucket_boundaries))}
            bucket_event_labels = {i: [] for i in range(len(bucket_boundaries))}
            bucket_attr_labels = {i: [] for i in range(len(bucket_boundaries))}
            for i, bucket_id in enumerate(bucket_ids):
                bucket_features[bucket_id].append(features[i][:bucket_input_sizes[bucket_id]])
                bucket_case_lengths[bucket_id].append(case_lengths[i])
                bucket_case_labels[bucket_id].append(case_labels[i])
                bucket_event_labels[bucket_id].append(event_labels[i])
                bucket_attr_labels[bucket_id].append(attr_labels[i])


            # Converting to np.arrays
            for bucket_id in bucket_features:
                bucket_features[bucket_id] = np.array(bucket_features[bucket_id])
                bucket_case_lengths[bucket_id] = np.array(bucket_case_lengths[bucket_id])
                bucket_case_labels[bucket_id] = np.array(bucket_case_labels[bucket_id])
                bucket_event_labels[bucket_id] = np.array(bucket_event_labels[bucket_id])
                bucket_attr_labels[bucket_id] = np.array(bucket_attr_labels[bucket_id])

            print('Finished generating models')
            return bucket_models, bucket_features, bucket_features, bucket_case_lengths, bucket_case_labels, bucket_event_labels, bucket_attr_labels

    @staticmethod
    def build_model(input_size, 
                    hidden_layers,
                    hidden_size_factor,
                    noise=None,
                    multi_task:SettingsMultiTask=None):
        
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import adam_v2

        print(f"Building model with input size: {input_size}")

        # Input layer
        input = Input(shape=(input_size,), name='input')
        x = input

        # Noise layer
        if noise is not None:
            x = GaussianNoise(noise)(x)

        # Hidden layers
        for i in range(hidden_layers):
            if isinstance(hidden_size_factor, list):
                factor = hidden_size_factor[i]
            else:
                factor = hidden_size_factor
            x = Dense(max(int(input_size * factor),64), activation='leaky_relu', name=f'hid{i + 1}')(x)
            x = Dropout(0.5)(x)

        # Output layer
        output = Dense(input_size, name='output')(x)
        # output = Dense(input_size, activation='sigmoid', name='output')(x)

        # Build model
        model = Model(inputs=input, outputs=output)

        if multi_task is not None:
            # Multi-task Loss Function
            attribute_perspectives = np.tile(multi_task.attribute_perspectives, [multi_task.max_length])
            attribute_types = np.tile(multi_task.attribute_types, [multi_task.max_length])
            attribute_splits = np.tile(multi_task.attribute_dims, [multi_task.max_length]) # np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
            # perspective_weights = [1,0,0,0] # Weights for each perspective, index based on Perspectives Enum
            loss = DAE.multi_task_loss(attribute_perspectives, attribute_types, attribute_splits, multi_task.perspective_weights)
        else:
            loss='mean_squared_error'

        # Compile model
        model.compile(
            optimizer=adam_v2.Adam(learning_rate=0.0001, beta_2=0.99, clipnorm=1.0),
            loss=loss
            # metrics=['accuracy']
        )

        return model

    @staticmethod
    def multi_task_loss(attribute_perspectives, attribute_types, attribute_splits, perspective_weights):
        def loss(y_true, y_pred):
            y_true_splits = tf.split(y_true, attribute_splits, axis=1)
            y_pred_splits = tf.split(y_pred, attribute_splits, axis=1)
            
            perspective_losses = defaultdict(list)
            for i in range(len(attribute_splits)):
                attribute_perspective = attribute_perspectives[i]
                attribute_type = attribute_types[i]

                y_true_split = y_true_splits[i]
                y_pred_split = y_pred_splits[i]

                # Based on what type of attribute it is determine which loss function to use
                if attribute_type == AttributeType.CATEGORICAL:
                    # If categorical preform a softmax
                    # RCVDB: TODO: Softmaxing does require the model to be able to predict if something is normal aka no anomalies
                    # Otherwise it will just artificially raise the y_pred while it should correctly predict all 0s

                    # y_pred_split = tf.nn.softmax(y_pred_split)
                    attribute_loss = tf.keras.losses.CategoricalCrossentropy()(y_true_split, y_pred_split)
                else: # attribute_type == AttributeType.NUMERICAL:
                    attribute_loss = tf.keras.losses.MeanSquaredError()(y_true_split, y_pred_split)
        
                # Based on which perspective the attribute belongs to 
                perspective_losses[attribute_perspective].append(attribute_loss)

            total_loss = 0
            for perspective_index, attribute_losses in perspective_losses.items():
                # Normalize for the number of attributes
                perspective_loss = tf.reduce_sum(attribute_losses) / len(attribute_losses)

                # Add the weight for the perspective
                total_loss += perspective_weights[perspective_index] * perspective_loss

            return total_loss
        return loss
  
    def train_and_predict(self, dataset:Dataset, batch_size=2, bucket_boundaries=None, categorical_encoding=EncodingCategorical.ONE_HOT, vector_size=None):
        model_buckets, features_buckets, targets_buckets, case_lengths_buckets, bucket_case_labels, bucket_event_labels, bucket_attr_labels = self.model_fn(dataset, bucket_boundaries, categorical_encoding, vector_size, **self.config)

        # Parameters
        attribute_dims = dataset.attribute_dims

        # If using WORD2VEC_ATC all categorical values are indexed first and then the numerical, internal order is maintained
        anomaly_perspectives = dataset.event_log.event_attribute_perspectives
        if categorical_encoding == EncodingCategorical.WORD_2_VEC_ATC:
            perspective_array = np.array(anomaly_perspectives)

            categorical_mask = np.array(dataset.attribute_types) == AttributeType.CATEGORICAL
            sorted_perspectives  = np.concatenate((perspective_array[categorical_mask], perspective_array[~categorical_mask]))

            anomaly_perspectives = sorted_perspectives

        bucket_trace_level_abnormal_scores = []
        bucket_event_level_abnormal_scores = [] 
        bucket_attr_level_abnormal_scores = []
        bucket_losses = []
        for i in range(len(model_buckets)):
            model = model_buckets[i]
            features = features_buckets[i]
            targets = targets_buckets[i]
            case_lengths = case_lengths_buckets[i]

            if bucket_boundaries is not None:
                # RCVDB: TODO Experimental removing start and end event in the encoding
                #case_max_length = bucket_boundaries[i] - 2
                case_max_length = bucket_boundaries[i]
            else:
                case_max_length = dataset.max_len

            # Setup the features and targets for online training
            feature_target_tf = tf.data.Dataset.from_tensor_slices((features, targets))
            feature_target_tf = feature_target_tf.batch(batch_size)
            total_steps = len(features) // batch_size

            losses=[]
            predictions=[]
            pbar = tqdm(enumerate(feature_target_tf), total=total_steps)
            for i, (x_batch, y_batch) in pbar:
                # RCVDB: Sanity check to see if the batches are valid:
                if len(x_batch) == 0 or len(y_batch) == 0:
                    print("Empty batch encountered!")
                if np.any(np.isnan(x_batch)) or np.any(np.isnan(y_batch)):
                    print("Batch contains NaNs!")

                # RCVDB: Train the model on the current batch and return the prediction
                # RCVDB: TODO: Tensorflow seems to throw optimisation warnings, suggesting that the code can be optimized. 

                # RCVDB: Goal is reconstruction, thus x and y are the same
                loss = model.train_on_batch(x_batch, y_batch)
                prediction_batch = model.predict_on_batch(x_batch)

                # RCVDB: Sanity check to see if the loss includes NanN
                if tf.reduce_any(tf.math.is_nan(loss)):
                    print("NaN detected in loss!")
                    raise ValueError("NaN detected in loss, stopping training!")

                losses.append(loss)
                for prediction in prediction_batch:
                    predictions.append(prediction)

                if (i+1) % 100 == 0 or i == 0:
                    pbar.set_postfix({'loss': loss})

            # (cases, events * flattened_attributes)
            errors_unmasked = np.power(targets - predictions, 2)
            # errors = targets - predictions

            # RCVDB: If trace2vec than the first vector_size of outputs can be discarded as it is not relevant for specific perspectives
            if categorical_encoding == EncodingCategorical.TRACE_2_VEC_ATC or categorical_encoding == EncodingCategorical.TRACE_2_VEC_C:
                errors_unmasked = errors_unmasked[:, vector_size:]

            # RCVDB: Mask empty events if no buckets are used or encoding method is not W2V
            if bucket_boundaries is None and categorical_encoding != EncodingCategorical.WORD_2_VEC_ATC: 
            # Applies a mask to remove the events not present in the trace   
            # (cases, flattened_errors) --> errors_unmasked
            # (cases, num_events) --> dataset.mask (~ inverts mask)
            # (cases, num_events, 1) --> expand dimension for broadcasting
            # (cases, num_events, attributes_dim) --> expand 2nd axis to size of the attributes
            # (cases, num_events * attributes_dim) = (cases, flattened_mask) --> reshape to match flattened error shape
                errors = errors_unmasked * np.expand_dims(~dataset.mask, 2).repeat(attribute_dims.sum(), 2).reshape(
                    dataset.mask.shape[0], -1)
            else:
                errors = errors_unmasked

            # If W2V all categorical events are embedded into a single vector
            if categorical_encoding == EncodingCategorical.WORD_2_VEC_ATC:
                attribute_type_counter = Counter(dataset.attribute_types)

                categorical_tiles = np.tile(vector_size, [attribute_type_counter[AttributeType.CATEGORICAL]])
                numerical_single_attribute = [1] * attribute_type_counter[AttributeType.NUMERICAL]
                numerical_tiles = np.tile(numerical_single_attribute, [case_max_length])
                w2v_tiles = np.concatenate((categorical_tiles, numerical_tiles))

                split_attribute = np.cumsum(w2v_tiles, dtype=int)[:-1]
            else:
                split_attribute = np.cumsum(np.tile(attribute_dims, [case_max_length]), dtype=int)[:-1]

            # Get the errors per attribute by splitting said trace
            # (attributes * events, cases, attribute_dimension)
            errors_attr_split = np.split(errors, split_attribute, axis=1)

            # Mean the attribute_dimension
            # Scalar attributes are left as is as they have a size of 1
            # np.mean for the proportion of the one-hot encoded predictions being wrong
            # np.sum for the total one-hot predictions being wrong
            # (attributes * events, cases)
            if categorical_encoding == EncodingCategorical.WORD_2_VEC_ATC:
                errors_attr_split_summed = [np.sum(attribute, axis=1) for attribute in errors_attr_split]
            else:
                errors_attr_split_summed = [np.mean(attribute, axis=1) for attribute in errors_attr_split]
            
            # Split the attributes based on which event it belongs to
            if categorical_encoding == EncodingCategorical.WORD_2_VEC_ATC:
                # Everything before the shared index is shared between each event as they are averaged during encoding
                shared_index = attribute_type_counter[AttributeType.CATEGORICAL]
                split_shared = np.split(errors_attr_split_summed, [shared_index])

                shared_indexes = split_shared[0]
                split_indexes = split_shared[1]

                # Combining both arrays
                expanded_shared_indexes = np.expand_dims(shared_indexes, axis=0) # Shape: (1, 5, 1000)
                # Duplicate the average error over all the events for compatibility with the rest of the code
                errors_event_split_categorical = np.tile(expanded_shared_indexes, (case_max_length, 1, 1)) # Shape: (13, 5, 1000)

                # Split the rest of the numerical values
                if attribute_type_counter[AttributeType.NUMERICAL] > 0:
                    event_numerical_size = attribute_type_counter[AttributeType.NUMERICAL]
                    split_event = np.arange(
                        start=event_numerical_size, #shared_index, 
                        stop=len(errors_attr_split_summed), 
                        step=event_numerical_size)[:-1]

                    errors_event_split_numerical_splits = np.split(split_indexes, split_event, axis=0)
                    errors_event_split_numerical = np.array(errors_event_split_numerical_splits)

                    errors_event_split = np.concatenate((errors_event_split_numerical, errors_event_split_categorical), axis=1)
                else:
                    errors_event_split = errors_event_split_categorical
                
                # (events, attributes, cases)
                # to
                # (attributes, events, cases)
                errors_event_split = np.transpose(errors_event_split, (1,0,2))
            else:
                split_event = np.arange(
                    start=case_max_length, 
                    stop=len(errors_attr_split_summed), 
                    step=case_max_length)
                
                # (attributes, events, cases)
                errors_event_split = np.split(errors_attr_split_summed, split_event, axis=0)

            # Split the attributes based on which perspective they belong to
            # (perspective, attributes, events, cases)
            grouped_error_scores_per_perspective = defaultdict(list)
            for event, anomaly_perspective in zip(errors_event_split, anomaly_perspectives):
                grouped_error_scores_per_perspective[anomaly_perspective].append(event)

            # Calculate the error proportions per the perspective per: attribute, event, trace
            trace_level_abnormal_scores = defaultdict(list)
            event_level_abnormal_scores = defaultdict(list) 
            attr_level_abnormal_scores = defaultdict(list)
            for anomaly_perspective in grouped_error_scores_per_perspective.keys():
                # Transpose the axis to make it easier to work with 
                # (perspective, attributes, events, cases)
                # to
                # (cases, events, attributes) 
                t = np.transpose(grouped_error_scores_per_perspective[anomaly_perspective], (2, 1, 0))
                event_dimension = case_lengths
                attribute_dimension = t.shape[-1]

                error_per_trace = np.sum(t, axis=(1,2)) / (event_dimension * attribute_dimension)
                trace_level_abnormal_scores[anomaly_perspective] = error_per_trace

                error_per_event = np.sum(t, axis=2) / attribute_dimension
                event_level_abnormal_scores[anomaly_perspective] = error_per_event

                error_per_attribute = t
                attr_level_abnormal_scores[anomaly_perspective] = error_per_attribute

            bucket_trace_level_abnormal_scores.append(trace_level_abnormal_scores)
            bucket_event_level_abnormal_scores.append(event_level_abnormal_scores)
            bucket_attr_level_abnormal_scores.append(attr_level_abnormal_scores)
            bucket_losses.append(losses)

        return bucket_trace_level_abnormal_scores, bucket_event_level_abnormal_scores, bucket_attr_level_abnormal_scores, bucket_losses, bucket_case_labels, bucket_event_labels, bucket_attr_labels
