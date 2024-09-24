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
from utils.enums import AttributeType, Heuristic, Perspective, Strategy, Mode, Base
from collections import defaultdict


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
                  noise=1) # None

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
    def model_fn(dataset:Dataset, **kwargs):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise
        from keras.models import Model
        from keras.optimizers import adam_v2

        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        noise = kwargs.pop('noise')

        features = dataset.flat_onehot_features_2d

        # Parameters
        input_size = features.shape[1]
        # print(f"Input Size: {input_size}")

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
            x = Dense(max(int(input_size * factor),64), activation='relu', name=f'hid{i + 1}')(x)
            x = Dropout(0.5)(x)

        # Output layer
        output = Dense(input_size, activation='sigmoid', name='output')(x)

        # Build model
        model = Model(inputs=input, outputs=output)

        multi_task = False
        if multi_task == True:
            # Multi-task Loss Function
            attribute_perspectives = np.tile(dataset.event_log.event_attribute_perspectives, [dataset.max_len])
            attribute_types = np.tile(dataset.attribute_types, [dataset.max_len])
            attribute_splits = np.tile(dataset.attribute_dims, [dataset.max_len]) # np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
            perspective_weights = [1,0,0,0] # Weights for each perspective, index based on Perspectives Enum
            loss = DAE.multi_task_loss(attribute_perspectives, attribute_types, attribute_splits, perspective_weights)
        else:
            loss='mean_squared_error'

        # Compile model
        model.compile(
            optimizer=adam_v2.Adam(learning_rate=0.0001, beta_2=0.99),
            loss=loss
            # metrics=['accuracy']
        )

        return model, features, features  # Features are also targets

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
  
    def train_and_predict(self, dataset:Dataset, batch_size=2):
        model, features, _ = self.model_fn(dataset, **self.config)
        
        dataset_tf = tf.data.Dataset.from_tensor_slices((features, features))
        dataset_tf = dataset_tf.batch(batch_size)
        total_steps = len(dataset.event_log) // batch_size

        losses=[]
        predictions=[]
        pbar = tqdm(enumerate(dataset_tf), total=total_steps)
        for i, (x_batch, y_batch) in pbar:
            # RCVDB: Train the model on the current batch and return the prediction
            # RCVDB: TODO: Tensorflow seems to throw optimisation warnings, suggesting that the code can be optimized. 

            # RCVDB: Goal is reconstruction, thus x and y are the same
            loss = model.train_on_batch(x_batch, y_batch)
            prediction_batch = model.predict_on_batch(x_batch)

            losses.append(loss)
            for prediction in prediction_batch:
                predictions.append(prediction)

            if (i+1) % 100 == 0 or i == 0:
                pbar.set_postfix({'loss': loss})

        # (cases, events * flattened_attributes)
        errors_unmasked = np.power(dataset.flat_onehot_features_2d - predictions, 2)

        # Applies a mask to remove the events not present in the trace   
        # (cases, flattened_errors) --> errors_unmasked
        # (cases, num_events) --> dataset.mask (~ inverts mask)
        # (cases, num_events, 1) --> expand dimension for broadcasting
        # (cases, num_events, attributes_dim) --> expand 2nd axis to size of the attributes
        # (cases, num_events * attributes_dim) = (cases, flattened_mask) --> reshape to match flattened error shape
        errors = errors_unmasked * np.expand_dims(~dataset.mask, 2).repeat(dataset.attribute_dims.sum(), 2).reshape(
            dataset.mask.shape[0], -1)
        
        # Get the split index of each attribute in each flattened trace
        split_attribute = np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]

        # Get the errors per attribute by splitting said trace
        # (attributes * events, cases, attribute_dimension)
        errors_attr_split = np.split(errors, split_attribute, axis=1)

        # Mean the attribute_dimension
        # Scalar attributes are left as is as they have a size of 1
        # np.mean for the proportion of the one-hot encoded predictions being wrong
        # np.sum for the total one-hot predictions being wrong
        # (attributes * events, cases)
        errors_attr_split_summed = [np.mean(attribute, axis=1) for attribute in errors_attr_split]
        
        # Split the attributes based on which event it belongs to
        split_event = np.arange(
            start=dataset.max_len, 
            stop=len(errors_attr_split_summed), 
            step=dataset.max_len)
        
        # (attributes, events, cases)
        errors_event_split = np.split(errors_attr_split_summed, split_event, axis=0)

        # Split the attributes based on which perspective they belong to
        # (perspective, attributes, events, cases)
        anomaly_perspectives = dataset.event_log.event_attribute_perspectives
        grouped_error_scores_per_perspective = defaultdict(list)
        for event, anomaly_perspectives in zip(errors_event_split, anomaly_perspectives):
            grouped_error_scores_per_perspective[anomaly_perspectives].append(event)

        # Calculate the error proportions per the perspective per: attribute, event, trace
        trace_level_abnormal_scores = defaultdict(list)
        event_level_abnormal_scores = defaultdict(list) 
        attr_level_abnormal_scores = defaultdict(list)
        for anomaly_perspective in grouped_error_scores_per_perspective.keys():
            # Transpose the axis to make it easier to work with 
            # (perspective, cases, events, attributes) 
            t = np.transpose(grouped_error_scores_per_perspective[anomaly_perspective], (2, 1, 0))
            event_dimension = dataset.case_lens
            attribute_dimension = t.shape[-1]

            error_per_trace = np.sum(t, axis=(1,2)) / (event_dimension * attribute_dimension)
            trace_level_abnormal_scores[anomaly_perspective] = error_per_trace

            error_per_event = np.sum(t, axis=2) / attribute_dimension
            event_level_abnormal_scores[anomaly_perspective] = error_per_event

            error_per_attribute = t
            attr_level_abnormal_scores[anomaly_perspective] = error_per_attribute

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores, losses
