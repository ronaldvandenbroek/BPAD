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

from collections import Counter
import gzip
import math
import os
import pickle as pickle

import numpy as np
import torch
from tqdm import tqdm
from utils.anomaly import label_to_targets
from utils.embedding.elmo import ProcessELMoEncoder
from utils.embedding.fixed_vector import FixedVectorEncoder
from utils.embedding.w2v import ProcessWord2VecEncoder
from utils.embedding.attribute_dictionary import AttributeDictionary
from utils.enums import AttributeType, EncodingCategorical, EncodingNumerical, Perspective
from utils.fs import EventLogFile
from processmining.event import Event
from processmining.case import Case
from processmining.log import EventLog


class Dataset(object):
    def __init__(self, 
                 dataset_name=None,
                 dataset_folder=None, 
                 beta=0, 
                 pretrain_percentage = 0,
                 vector_size = 50,
                 window_size = 10, 
                 prefix=True, 
                 categorical_encoding=EncodingCategorical.ONE_HOT,
                 numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                 fs_save=None):
        # Public properties
        if dataset_folder:
            self.dataset_name = os.path.join(dataset_folder, dataset_name)
        else:
            self.dataset_name = dataset_name

        self.beta=beta   #used by GAMA
        self.attribute_types = None
        self.attribute_keys = None
        self.fs_save=fs_save

        # Encoding types
        self.categorical_encoding = categorical_encoding
        self.numerical_encoding = numerical_encoding

        # RCVDB: Renaming classes/labels to labels per abstraction level
        self.attr_labels = None
        self.event_labels = None
        self.case_labels = None

        self.encoders = None
        self.trace_graphs = []
        self.trace_graphs_GAE = []
        self.node_dims=[]
        self.edge_indexs = []
        self.node_xs = []
        self.pretrain_percentage = pretrain_percentage  
        self.vector_size = vector_size
        self.window_size = window_size

        # Private properties
        self._mask = None
        self._attribute_dims = None
        self._case_lens = None
        self._features = None
        self._event_log = None

        # Load dataset
        if self.dataset_name is not None:
            self.load(self.dataset_name, prefix)

    def load(self, dataset_name, prefix):
        """
        Load dataset from disk. If there exists a cached file, load from cache. If no cache file exists, load from
        Event Log and cache it.

        :param dataset_name:
        :return:
        """
        el_file = EventLogFile(dataset_name)
        self.dataset_name = el_file.name
        
        # RCVDB: Skipping caching TODO reimplement
        # # Check for cache
        # if el_file.cache_file.exists():
        #     self._load_dataset_from_cache(el_file.cache_file)
        #     self._gen_trace_graphs()
        #     self._gen_trace_graphs_GAE()


        # Else generator from event log
        if el_file.path.exists():
            self._event_log = EventLog.load(el_file.path, prefix)
            self.from_event_log(self._event_log)
            # RCVDB: Skipping caching and generating graphs TODO reimplement
            # self._cache_dataset(el_file.cache_file)
            # self._gen_trace_graphs()
            # self._gen_trace_graphs_GAE()
        else:
            print(el_file.path)
            raise FileNotFoundError()

    def _load_dataset_from_cache(self, file):
        with gzip.open(file, 'rb') as f:
            (self._features, self.classes, self.labels, self._case_lens, self._attribute_dims,
             self.encoders, self.attribute_types, self.attribute_keys) = pickle.load(f)

    def _cache_dataset(self, file):
        with gzip.open(file, 'wb') as f:
            pickle.dump((self._features, self.classes, self.labels, self._case_lens, self._attribute_dims,
                         self.encoders, self.attribute_types, self.attribute_keys), f)

    def __len__(self):
        return self.num_cases
    
    @property
    def mask(self):
        if self._mask is None:
            self._mask = np.ones(self._features[0].shape, dtype=bool)
            for m, j in zip(self._mask, self.case_lens):
                m[:j] = False
        return self._mask

    @property
    def event_log(self):
        """Return the event log object of this dataset."""
        if self.dataset_name is None:
            raise ValueError(f'dataset {self.dataset_name} cannot be found')

        if self._event_log is None:
            self._event_log = EventLog.load(self.dataset_name)
        return self._event_log

    @property
    def text_labels(self):
        """Return the labels transformed into text, one string for each case in the event log."""
        return np.array(['Normal' if l == 'normal' else l['anomaly'] for l in self.labels])

    @property
    def unique_text_labels(self):
        """Return unique text labels."""
        return sorted(set(self.text_labels))

    @property
    def unique_anomaly_text_labels(self):
        """Return only the unique anomaly text labels."""
        return [l for l in self.unique_text_labels if l != 'Normal']

    @property
    def case_target(self):
        z = np.zeros(self.num_cases)
        for i in self.anomaly_indices:
            z[i] = 1
        return z

    @property
    def normal_indices(self):
        return self.get_indices_for_type('Normal')

    @property
    def cf_anomaly_indices(self):
        if len(self.text_labels) > 0:
            return np.where(np.logical_and(self.text_labels != 'Normal', self.text_labels != 'Attribute'))[0]
        else:
            return range(int(self.num_cases))

    @property
    def anomaly_indices(self):
        if len(self.text_labels) > 0:
            return np.where(self.text_labels != 'Normal')[0]
        else:
            return range(int(self.num_cases))

    @property
    def case_lens(self):
        """Return length for each case in the event log as 1d NumPy array."""
        return self._case_lens

    @property
    def attribute_dims(self):
        """Return dimensionality of attributes from event log."""
        if self._attribute_dims is None:
            self._attribute_dims = np.asarray([f.max() if t == AttributeType.CATEGORICAL else 1 for f, t in
                                               zip(self._features, self.attribute_types)])
        # print(f'Attribute Dimensions: {self._attribute_dims}')
        return self._attribute_dims
    
    @property
    def num_attributes(self):
        """Return the number of attributes in the event log."""
        return len(self.features)

    @property
    def num_cases(self):
        """Return number of cases in the event log, i.e., the number of examples in the dataset."""
        return len(self.features[0])

    @property
    def num_events(self):
        """Return the total number of events in the event log."""
        return sum(self.case_lens)

    @property
    def max_len(self):
        """Return the length of the case with the most events."""
        return self.features[0].shape[1]
    
    # RCVDB: This version of the function aims for the models to predict the anomaly perspective themselves, 
    # however this can also be achieved by generating an error score for each attribute 
    # and depending on where the error is the anomaly perspectives can be determined
    @staticmethod
    def _get_classes_and_labels_from_event_log(event_log):
        """
        Extract anomaly labels from event log format and transform into anomaly detection targets.

        :param event_log:
        :return:
        """
        # RCVDB: Create a target matix for multi-perspective anomaly detection
        # +1 for end event and +1 for start event
        num_events = event_log.max_case_len + 2
        num_attributes = event_log.num_event_attributes
        num_perspectives = len(Perspective.keys())
        num_cases = len(event_log.cases)

        # Labels One-hot encoded per case, event and attribute
        labels_event_level = np.zeros((num_cases, num_events, num_perspectives), dtype=bool)
        labels_case_level = np.zeros((num_cases, num_perspectives), dtype=bool)
        labels_attr_level = []

        for case_index, case in enumerate(event_log):
            case:Case
            case_targets = np.zeros((num_events, num_attributes, num_perspectives), dtype=bool) 
            for event_index, event in enumerate(case.events):
                event:Event
                if '_label' in event.attributes and event.attributes['_label'] is not None:
                    event_labels = event.attributes['_label']
                    # if case_index < 2:
                    #     print(f'{case_index} {event_index} {event_labels}')
                    for label in event_labels:
                        # Encode the label into the targets:
                        case_targets, perspective = label_to_targets(event_log, case_targets, event_index, label)

                        # Debug code to see if labels are set correctly TODO can remove if not nessesary anymore
                        # if case_index < 2:
                        #     print(f' Setting value at {case_index},{event_index},{perspective}')

                        # Encode the perspective into labels
                        if perspective is not None: # If perspective is none then no anomalies need to be registered
                            labels_case_level[case_index, perspective] = True
                            labels_event_level[case_index, event_index, perspective] = True

            # Create a list of labels per case
            labels_attr_level.append(case_targets)

        # Should result in a 4d tensor of size (num_cases, num_events, num_attributes, num_perspectives)
        attr_labels = np.asarray(labels_attr_level)
        print(f'Attribute Labels shape: {attr_labels.shape}')

        # Should result in a 3d tensor of size (num_cases, num_events, num_perspectives)
        event_labels = np.asarray(labels_event_level)
        print(f'Event Labels shape: {event_labels.shape}')

        # Should result in a 2d tensor of size (num_cases, num_perspectives)
        # Was initially a list of strings with one label per case, now it represents all perspectives present per event in each case
        case_labels = np.asarray(labels_case_level)
        print(f'Case Labels shape: {case_labels.shape}')

        # RCVDB: TODO remove debug prints after label setting has be confirmed to be correct
        # print(f'Example target of event: \n {targets[0]}')
        # print(f'Example labels of cases:')
        # for i in range(5):
        #     print(f'Case {i}: {labels[i]}')

        return attr_labels, event_labels, case_labels

    def _from_event_log(self, event_log:EventLog, include_attributes=None):
        """
        Transform event log as feature columns.

        Categorical attributes are integer encoded. Shape of feature columns is
        (number_of_cases, max_case_length, number_of_attributes).

        :param include_attributes:

        :return: feature_columns, case_lens
        """
        if include_attributes is None:
            include_attributes = event_log.event_attribute_keys

        feature_columns = dict(name=[])
        case_lens = []
        attr_types = event_log.get_attribute_types(include_attributes)

        # Create beginning of sequence event
        start_event = dict((a, EventLog.start_symbol if t == AttributeType.CATEGORICAL else 0.0) for a, t in
                           zip(include_attributes, attr_types))
        start_event = Event(timestamp=None, **start_event)

        # Create end of sequence event
        end_event = dict((a, EventLog.end_symbol if t == AttributeType.CATEGORICAL else 0.0) for a, t in
                         zip(include_attributes, attr_types))
        end_event = Event(timestamp=None, **end_event)

        # Save all values in a flat 1d array. This is necessary for the preprocessing. We will reshape later.
        for i, case in enumerate(event_log.cases):
            # RCVDB: Do not add the end event if the trace is not 'complete' or in other words is a prefix
            case:Case

            if case.complete_trace: # or not case.complete_trace:
                case_lens.append(case.num_events + 2)  # +2 for start and end events
                trace = [start_event] + case.events + [end_event]
            else:
                case_lens.append(case.num_events + 1)  # +1 for start event
                trace = [start_event] + case.events

            for event in trace:
                for attribute in event_log.event_attribute_keys:
                    # Get attribute value from event log
                    if attribute == 'name':
                        attr = event.name
                    elif attribute in include_attributes:
                        attr = event.attributes[attribute]
                    else:
                        # Ignore the attribute name because its not part of included_attributes
                        continue

                    # Add to feature columns
                    if attribute not in feature_columns.keys():
                        feature_columns[attribute] = []
                    feature_columns[attribute].append(attr)

        # Data preprocessing
        final_attribute_types = []
        encoders = {}
        dictionary_starting_index = 3 # 0,1,2 reserved for padding,start and endsymbol respectively
        for index, (key, attribute_type) in enumerate(zip(feature_columns.keys(), attr_types)):
            replace_attribute_type = None
            # print(f'Pre-Processing {key} {attribute_type}')

            # Integer encode categorical data
            if attribute_type == AttributeType.CATEGORICAL:# and not self.categorical_encoding == EncodingCategorical.ELMO:
                # Dynamic max size
                unknown_buffer_percentage = 1.25
                unique_values_count = len(set(feature_columns[key]))
                dictionary_size = math.ceil(unique_values_count * unknown_buffer_percentage)
                encoder = AttributeDictionary(max_size=dictionary_size, 
                                              start_index=dictionary_starting_index,
                                              start_symbol=event_log.start_symbol,
                                              end_symbol=event_log.end_symbol)
                
                # print(f'Encoding {key} with dictionary size {dictionary_size}')
                feature_columns[key] = encoder.encode_list(feature_columns[key])
                # print(encoder.encodings.keys())
                encoders[key] = encoder
                
                if self.categorical_encoding != EncodingCategorical.TOKENIZER:
                    # To ensure that every attribute value has an unique token ensure that the starting index starts outside of the buffer
                    # Otherwise with Word2Vec attribute integers will conflict
                    # This is not relevant when using a Tokenizer for the Transformer
                    dictionary_starting_index += dictionary_size

                # If categorical encoding is none then categorical values are treated as numerical thus can scale
                if self.categorical_encoding == EncodingCategorical.NONE:# or self.categorical_encoding == EncodingCategorical.FIXED_VECTOR:
                    if self.numerical_encoding == EncodingNumerical.MIN_MAX_SCALING:
                        feature_columns[key] = self._min_max_scaling(feature_columns[key])
                    replace_attribute_type = AttributeType.NUMERICAL

            # Normalize numerical data
            elif attribute_type == AttributeType.NUMERICAL:
                # print(self.numerical_encoding, EncodingNumerical.MIN_MAX_SCALING)
                if self.numerical_encoding == EncodingNumerical.MIN_MAX_SCALING:
                    # print('Minmax scaling ' + key)
                    feature_columns[key] = self._min_max_scaling(feature_columns[key])

            if replace_attribute_type is None:
                final_attribute_types.append(attribute_type)
            else:
                final_attribute_types.append(replace_attribute_type)

        # Transform back into sequences
        case_lens = np.array(case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        # if self.categorical_encoding == EncodingCategorical.ELMO:
        #     features = [np.zeros((case_lens.shape[0], case_lens.max()),dtype=str) for _ in range(len(feature_columns))]
        # else:
        features = [np.zeros((case_lens.shape[0], case_lens.max()),dtype=np.float32) for _ in range(len(feature_columns))]

        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            for k, key in enumerate(feature_columns):
                x = feature_columns[key]
                features[k][i, :case_len] = x[offset:offset + case_len]

        # RCVDB: Debug prints to showcase the encoded labels
        print(f'Feature Columns: {feature_columns.keys()}')
        print(f'Feature Shape: {features[0].shape}')
        # print(f'Example feature over multiple cases:')
        # for i in range(5):
        #     print(f'Case {i} Features, name: {features[0][i]}, arrival-time: {features[1][i]}, user: {features[-1][i]}')
        
        # example_case = 0
        # print(f'All feature over a single case {example_case}:')
        # for index, key in enumerate(feature_columns.keys()):
        #     print(f'{key} \t {features[index][example_case]}')    
        print(f'Case Length: {case_lens}')
        print(f'Attribute Types: {final_attribute_types}')
        print(f'Encoders: {encoders}')

        return features, case_lens, final_attribute_types, encoders

    def from_event_log(self, event_log):
        """
        Load event log file and set the basic fields of the `Dataset` class.

        :param event_log: event log name as string
        :return:
        """
        # Get features from event log
        self._features, self._case_lens, self.attribute_types, self.encoders = self._from_event_log(event_log)
        print("Generating log from data.")
        
        # Get targets and labels from event log
        self.attr_labels, self.event_labels, self.case_labels = self._get_classes_and_labels_from_event_log(event_log)
        print("Generating labels from log.")  

        # Attribute keys (names)
        # self.attribute_keys = [a.replace(' ', '_') for a in self.event_log.event_attribute_keys]
        self.attribute_keys = [a.replace(':', '_').replace(' ', '_') for a in self.event_log.event_attribute_keys]
        print("Generating attribute keys from log.")  

    def assign_to_buckets(self, bucket_boundaries):
        max_bucket_size = len(bucket_boundaries)
        bucket_ids = []
        for length in self.case_lens:
            # If the case is larger than the boundaries add it to the overflow bucket
            if length > bucket_boundaries[-1]:
                bucket_ids.append(max_bucket_size)

            # Determine the bucket the case fits in based on the bucket boundaries
            for i, boundary in enumerate(bucket_boundaries):
                if length <= boundary:
                    bucket_ids.append(i)
                    break

        return bucket_ids        

    def _to_categorical(self, y, num_classes=None, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
    
    def _min_max_scaling(self, feature_column):
        # RCVDB: Implementing min-max scaling to avoid relying on a normality assumption as experiments have shown that a normal distribution cannot be assumed in most features
        # feature_columns[key] = (f - f.mean()) / f.std()  # 0 mean and 1 std normalization
        f_min = np.min(feature_column)
        f_max = np.max(feature_column)
        # print(f_min, f_max)
        if f_max > f_min:
            scaled_feature_column = (feature_column - f_min) / (f_max - f_min)

            f_min = np.min(scaled_feature_column)
            f_max = np.max(scaled_feature_column)
            # print(f_min, f_max)
            return scaled_feature_column
        else:
            return np.zeros_like(feature_column)

    def attribute_type_count(self, type=AttributeType):
        counter = Counter(self.attribute_types)
        return counter[type]

    def get_indices_for_type(self, t):
        if len(self.text_labels) > 0:
            return np.where(self.text_labels == t)[0]
        else:
            return range(int(self.num_cases))

    @staticmethod
    def remove_time_dimension(x):
        return x.reshape((x.shape[0], np.product(x.shape[1:])))

    @property
    def _reverse_features(self):
        reverse_features = [np.copy(f) for f in self._features]
        for f in reverse_features:
            for _f, m in zip(f, self.mask):
                _f[~m] = _f[~m][::-1]
        return reverse_features

    # Features
    @property
    def features(self):
        return self._features

    @property
    def flat_features(self):
        """
        Return combined features in one single tensor.

        `features` returns one tensor per attribute. This method combines all attributes into one tensor. Resulting
        shape of the tensor will be (number_of_cases, max_case_length, number_of_attributes).

        :return:
        """
        return np.dstack(self.features)

    @property
    def flat_features_2d(self):
        """
        Return 2d tensor of flat features.

        Concatenates all attributes together, removing the time dimension. Resulting tensor shape will be
        (number_of_cases, max_case_length * number_of_attributes).

        :return:
        """
        return self.remove_time_dimension(self.flat_features)

    @property
    def binary_targets(self):
        """Return targets for anomaly detection; 0 = normal, 1 = anomaly."""
        # RCVDB: With multi-class prediction, the targets are already binary
        print(f'Binary target shape: {self.classes.shape}')
        return self.classes

        # if self.classes is not None and len(self.classes) > 0:
        #     print(f'Classes/Targets shape: {self.classes.shape}')
        #     print(f'Labels shape: {self.labels.shape}')
        #     targets = np.copy(self.classes)
        #     targets[targets > Class.ANOMALY] = Class.ANOMALY
        #     return targets
        # return None

    def _build_w2v_model(self):
        return ProcessWord2VecEncoder(
            encoders=self.encoders,
            pretrain_percentage=self.pretrain_percentage,
            attribute_types=self.attribute_types,
            event_attribute_keys=self.event_log.event_attribute_keys,
            features=self._features,
            event_log=self.event_log,
            vector_size=self.vector_size,
            window=self.window_size,
            fs_save=self.fs_save)         

    def flat_w2v_features_2d_average(self, trace2vec=False):
        w2v_encoder = self._build_w2v_model()
        features, self._attribute_dims = w2v_encoder.encode_flat_features_2d_average(trace2vec=trace2vec)
        return features
        
    def flat_w2v_features_2d(self, trace2vec=False):
        w2v_encoder = self._build_w2v_model()
        features, self._attribute_dims = w2v_encoder.encode_flat_features_2d(attribute_keys=self.event_log.event_attribute_keys, trace2vec=trace2vec)
        return features
    
    def flat_fixed_vector_features_2d(self):
        fixed_vector_encoder = FixedVectorEncoder(
            vector_size=self.vector_size, 
            features=self.features,
            attribute_types=self.attribute_types,
            event_attribute_keys=self.event_log.event_attribute_keys)
        features, self._attribute_dims = fixed_vector_encoder.flat_features_2d()
        return features
    
    def flat_elmo_features_2d(self):
        elmo_encoder = ProcessELMoEncoder(
            vector_size=self.vector_size, 
            features=self.features)
        self._attribute_dims = np.array([self.vector_size] * len(self.attribute_dims))
        return elmo_encoder.flat_elmo_features_2d()

    # Embedding Features
    @property
    def embedding_features(self):
        import torch.nn as nn

        embedded_features = []

        for feature, attribute_type in zip(self._features, self.attribute_types):
            embedding_dim = 5  # Fixed-size output vector

            unique_features = np.unique(feature)
            if attribute_type == AttributeType.CATEGORICAL:
                num_categories = len(unique_features)
            elif attribute_type == AttributeType.NUMERICAL:
                num_categories = np.max(unique_features)

            print(num_categories)
            attribute_embedding_layer = nn.Embedding(num_categories, embedding_dim)

            try:
                feature_tensor = torch.tensor(feature).long()
                feature_embedding = attribute_embedding_layer(feature_tensor).detach().numpy()
                embedded_features.append(feature_embedding)
            except:
                print('test')

        return embedded_features
    
    @property
    def flat_embedding_features(self):
        flat_embedding_features = np.concatenate(self.embedding_features, axis=2)

        return flat_embedding_features
    
    @property
    def flat_embedding_features_2d(self):
        flat_embedding_features_2d = self.remove_time_dimension(self.flat_embedding_features)
        print('Flat-Embedding features 2d shape: ', len(flat_embedding_features_2d), len(flat_embedding_features_2d[0]))

        return flat_embedding_features_2d

    # One-hot Features
    @property
    def onehot_features(self):
        """
        Return one-hot encoding of integer encoded features, while numerical features are passed as they are.

        As `features` this will return one tensor for each attribute. Shape of tensor for each attribute will be
        (number_of_cases, max_case_length, attribute_dimension). The attribute dimension refers to the number of unique
        values of the respective attribute encountered in the event log.

        :return:
        """
        # one_hot_features = [self._to_categorical(f)[:, :, 1:] if t == AttributeType.CATEGORICAL else np.expand_dims(f, axis=2)
        #         for f, t in zip(self._features, self.attribute_types)]

        # RCVDB: With the real world dataset the above operation causes memory errors
        # Setting dtype for numerical stability and memory efficiency
        dtype = np.float32

        # Generator expression to generate features on-demand
        one_hot_features = (
            (self._to_categorical(f).astype(dtype)[:, :, 1:] if t == AttributeType.CATEGORICAL 
            else np.expand_dims(f.astype(dtype), axis=2))
            for f, t in zip(self._features, self.attribute_types)
        )

        # If a list is required, wrap the generator expression in list()
        one_hot_features = list(one_hot_features)
        
        # RCVDB: Tensor seems to be of shape (attribute_dimension, number_of_cases, max_case_length)
        print('One-hot features shape: ', len(one_hot_features), len(one_hot_features[0]), len(one_hot_features[0][0]))

        # RCVDB: Debug to get an overview of how the features are encoded
        # for i in range(len(one_hot_features)):
        #     print(one_hot_features[i][0][0])

        return one_hot_features

    @property
    def flat_onehot_features(self):
        """
        Return combined one-hot features in one single tensor.

        One-hot vectors for each attribute in each event will be concatenated. Resulting shape of tensor will be
        (number_of_cases, max_case_length, attribute_dimension[0] + attribute_dimension[1] + ... + attribute_dimension[n]).

        :return:
        """
        # flat_one_hot_features = np.concatenate(self.onehot_features, axis=2)
        # RCVDB: With the real world dataset the above operation causes memory errors
        # Setting dtype for numerical stability and memory efficiency

        flat_one_hot_features = np.concatenate([feature for feature in self.onehot_features], axis=2)
        print('Flat-One-hot features shape: ', len(flat_one_hot_features), len(flat_one_hot_features[0]), len(flat_one_hot_features[0][0]))
        # print(flat_one_hot_features.shape)

        return flat_one_hot_features

    @property
    def flat_onehot_features_2d(self):
        """
        Return 2d tensor of one-hot encoded features.

        Same as `flat_onehot_features`, but with flattened time dimension (the second dimension). Resulting tensor shape
        will be (number_of_cases, max_case_length * (attribute_dimension[0] + attribute_dimension[1] + ... + attribute_dimension[n]).

        :return:
        """
        flat_onehot_features_2d = self.remove_time_dimension(self.flat_onehot_features)
        print('Flat-One-hot features 2d shape: ', len(flat_onehot_features_2d), len(flat_onehot_features_2d[0]))

        return flat_onehot_features_2d
    
    @property
    def onehot_train_targets(self):
        """
        Return targets to be used when training predictive anomaly detectors.

        Returns for each case the case shifted by one event to the left. A predictive anomaly detector is trained to
        predict the nth + 1 event of a case when given the first n events.

        :return:
        """
        return [np.pad(f[:, 1:], ((0, 0), (0, 1), (0, 0)), mode='constant') if t == AttributeType.CATEGORICAL else f
                for f, t in zip(self.onehot_features, self.attribute_types)]