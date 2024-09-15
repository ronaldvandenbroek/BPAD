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

from baseline.binet.core import NNAnomalyDetector
from utils.dataset import Dataset
from utils.enums import Heuristic, Strategy, Mode, Base
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
                  noise=None)

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

        # Compile model
        model.compile(
            optimizer=adam_v2.Adam(learning_rate=0.0001, beta_2=0.99),
            loss='mean_squared_error',
        )

        return model, features, features  # Features are also targets

    def detect(self, dataset:Dataset):
        """
        Calculate the anomaly score for each event attribute in each trace.
        Anomaly score here is the mean squared error.

        :param traces: traces to predict
        :return:
            scores: anomaly scores for each attribute;
                            shape is (#traces, max_trace_length - 1, #attributes)

        """
        # Get features
        _, features, _ = self.model_fn(dataset, **self.config)

        # Parameters
        input_size = int(self.model.input.shape[1])
        features_size = int(features.shape[1])
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)], mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        predictions=[]
        batch_size = 128
        i=0
        while features.shape[0]>=batch_size*i:
            predictions.append( self.model.predict(features[batch_size*i:batch_size*(i+1)], verbose=True))
            i += 1

        predictions= np.concatenate(predictions, 0)

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

        return trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores
