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
from novel.utils.component_runtime_tracker import ComponentRuntimeTracker
from novel.utils.iteration_runtime_tracker import IterationRuntimeTracker
from utils import process_results
from utils.dataset import Dataset
from utils.enums import AttributeType, EncodingCategorical, Heuristic, Strategy, Mode, Base
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
                  hidden_size_factor=.01, # .005, # .01,
                  noise=None # 0.5
     ) 

    def __init__(self, config):
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
        super(DAE, self).__init__(config)

        # Update the config with the additional parameters
        self.config.update(config)

    def model_fn(self, dataset:Dataset, bucket_boundaries:list, categorical_encoding, vector_size=None):
        component_runtime_tracker:ComponentRuntimeTracker = self.config.get('component_runtime_tracker', None)
        component_runtime_tracker.start_component('prepare_data')

        hidden_layers = self.config.get('hidden_layers')
        hidden_size_factor = self.config.get('hidden_size_factor')
        noise = self.config.get('noise')

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

        component_runtime_tracker.end_component('prepare_data')

        component_runtime_tracker.start_component('build_model')

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

            component_runtime_tracker.end_component('build_model')
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

        # model.summary()

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
  
    def train_and_predict(self, dataset:Dataset):
        bucket_boundaries = self.config.get('bucket_boundaries', None)
        categorical_encoding = self.config.get('categorical_encoding', EncodingCategorical.ONE_HOT)
        vector_size = self.config.get('vector_size', None)
        batch_size = self.config.get('batch_size', 2)
        online_training = self.config.get('online_training', True)
        component_runtime_tracker:ComponentRuntimeTracker = self.config.get('component_runtime_tracker', None)

        model_buckets, features_buckets, targets_buckets, case_lengths_buckets, bucket_case_labels, bucket_event_labels, bucket_attr_labels = self.model_fn(dataset, bucket_boundaries, categorical_encoding, vector_size)

        # Parameters
        attribute_dims = dataset.attribute_dims

        # If using ATC all categorical values are indexed first and then the numerical, internal order is maintained
        attribute_perspectives = dataset.event_log.event_attribute_perspectives
        attribute_perspectives_original = dataset.event_log.event_attribute_perspectives 
        attribute_names = dataset.event_log.event_attribute_keys
        attribute_names_original = dataset.event_log.event_attribute_keys
        if categorical_encoding in (EncodingCategorical.WORD_2_VEC_ATC, EncodingCategorical.TRACE_2_VEC_ATC):
            categorical_mask = np.array(dataset.attribute_types) == AttributeType.CATEGORICAL

            perspective_array = np.array(attribute_perspectives)
            sorted_perspectives  = np.concatenate((perspective_array[categorical_mask], perspective_array[~categorical_mask]))
            attribute_perspectives = sorted_perspectives

            name_array = np.array(attribute_names)
            sorted_names = np.concatenate((name_array[categorical_mask], name_array[~categorical_mask]))
            attribute_names = sorted_names

            # self._attribute_dims = np.array([self.vector_size] * len(self.attribute_dims))

        runtime_tracker = IterationRuntimeTracker(batch_size=self.config.get('batch_size'))
        runtime_tracker_inference = IterationRuntimeTracker(batch_size=self.config.get('batch_size'))
        runtime_tracker_train = IterationRuntimeTracker(batch_size=self.config.get('batch_size'))
        runtime_tracker_save_results = IterationRuntimeTracker(batch_size=self.config.get('batch_size'))
        
        bucket_trace_level_abnormal_scores = []
        bucket_event_level_abnormal_scores = [] 
        bucket_attr_level_abnormal_scores = []
        bucket_errors_raw = []
        bucket_losses = []
        for i in range(len(model_buckets)):
            component_runtime_tracker.start_component('train_predict_model')

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
                runtime_tracker.start_iteration()

                # RCVDB: Sanity check to see if the batches are valid:
                if len(x_batch) == 0 or len(y_batch) == 0:
                    print("Empty batch encountered!")
                if np.any(np.isnan(x_batch)) or np.any(np.isnan(y_batch)):
                    print("Batch contains NaNs!")

                # RCVDB: Train the model on the current batch and return the prediction
                # RCVDB: TODO: Tensorflow seems to throw optimisation warnings, suggesting that the code can be optimized. 

                # RCVDB: Goal is reconstruction, thus x and y are the same
                runtime_tracker_train.start_iteration()
                loss = model.train_on_batch(x_batch, y_batch)
                runtime_tracker_train.end_iteration()

                if online_training:
                    runtime_tracker_inference.start_iteration()
                    prediction_batch = model.predict_on_batch(x_batch)
                    runtime_tracker_inference.end_iteration()

                    runtime_tracker_save_results.start_iteration()
                    for prediction in prediction_batch:
                        predictions.append(prediction)
                    runtime_tracker_save_results.end_iteration()

                runtime_tracker.end_iteration()

                # RCVDB: Sanity check to see if the loss includes NanN
                if tf.reduce_any(tf.math.is_nan(loss)):
                    print("NaN detected in loss!")
                    raise ValueError("NaN detected in loss, stopping training!")
                losses.append(loss)

                if (i+1) % 100 == 0 or i == 0:
                    pbar.set_postfix({'loss': loss})

            if not online_training:
                pbar = tqdm(enumerate(feature_target_tf), total=total_steps)
                for i, (x_batch, y_batch) in pbar:
                    runtime_tracker_inference.start_iteration()
                    prediction_batch = model.predict_on_batch(x_batch)
                    runtime_tracker_inference.end_iteration()
                    
                    runtime_tracker_save_results.start_iteration()
                    for prediction in prediction_batch:
                        predictions.append(prediction)
                    runtime_tracker_save_results.end_iteration()

            component_runtime_tracker.end_component('train_predict_model')
            component_runtime_tracker.start_component('post_process_results')

            # (cases, events * flattened_attributes)
            errors_raw = targets - predictions

            bucket_errors_raw.append(errors_raw)
            bucket_losses.append(losses)

            trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = process_results.process_bucket_results(
                errors_raw=errors_raw,
                categorical_encoding=categorical_encoding,
                vector_size=vector_size,
                bucketing=bucket_boundaries is not None,
                attribute_dims=attribute_dims,
                dataset_mask=dataset.mask,
                attribute_types=dataset.attribute_types,
                case_max_length=case_max_length,
                anomaly_perspectives=attribute_perspectives,
                case_lengths=case_lengths,
                error_power=2
            )

            bucket_trace_level_abnormal_scores.append(trace_level_abnormal_scores)
            bucket_event_level_abnormal_scores.append(event_level_abnormal_scores)
            bucket_attr_level_abnormal_scores.append(attr_level_abnormal_scores)

            component_runtime_tracker.end_component('post_process_results')

        
        runtime_results_all = runtime_tracker.get_average_std_run_time()
        runtime_results_inference = runtime_tracker_inference.get_average_std_run_time()
        # runtime_results_loss = runtime_tracker_loss.get_average_std_run_time()
        runtime_results_train = runtime_tracker_train.get_average_std_run_time()
        runtime_tracker_save_results = runtime_tracker_save_results.get_average_std_run_time()
        runtime_results = {
            'overall_runtimes': runtime_results_all, 
            'inference_runtimes': runtime_results_inference,
            'loss_runtimes': None,
            'train_runtimes': runtime_results_train,
            'save_results_runtimes': runtime_tracker_save_results
        }

        return (bucket_trace_level_abnormal_scores, 
                bucket_event_level_abnormal_scores, 
                bucket_attr_level_abnormal_scores, 
                bucket_losses, 
                bucket_case_labels, 
                bucket_event_labels, 
                bucket_attr_labels, 
                bucket_errors_raw,
                attribute_perspectives,
                attribute_perspectives_original,
                attribute_names,
                attribute_names_original,
                None,
                None,
                runtime_results)
