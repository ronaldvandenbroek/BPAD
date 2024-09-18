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

import gzip
import pickle as pickle

import numpy as np
import torch
from torch_geometric.data import Data
from utils.anomaly import label_to_targets
from utils.enums import AttributeType, Perspective
from utils.enums import Class
from utils.enums import PadMode
from utils.fs import EventLogFile
from processmining.event import Event
from processmining.case import Case
from processmining.log import EventLog


class Dataset(object):
    def __init__(self, dataset_name=None, beta=0, label_percent = 0, prefix=True):
        # Public properties
        self.dataset_name = dataset_name
        self.beta=beta   #used by GAMA
        self.attribute_types = None
        self.attribute_keys = None

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
        self.label_percent = label_percent   # Used by weakly supervised methods to control the percentage of anomalies labeled during training

        # Private properties
        self._mask = None
        self._attribute_dims = None
        self._case_lens = None
        self._features = None
        self._event_log = None


        # Load dataset
        if self.dataset_name is not None:
            self.load(self.dataset_name, prefix)

        # RCVDB: TODO Support the weakly supervised methods
        # self.labeled_indices = np.random.choice(self.anomaly_indices, size=max(int(
        #     len(self.anomaly_indices) * self.label_percent),1), replace=False)  ### Used by weakly supervised methods to indicate indices of labeled anomalies during training

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
            raise FileNotFoundError()

    def _gen_trace_graphs(self):

        graph_relation = np.zeros((self.attribute_dims[0]+1,self.attribute_dims[0]+1),dtype='int32')
        for case_index in range(self.num_cases):
            if self.case_lens[case_index]>1:
                for activity_index in range(1, self.case_lens[case_index]):
                    graph_relation[ self.features[0][case_index][activity_index - 1] , self.features[0][case_index][activity_index] ] += 1
        dims_temp = []
        dims_temp.append(self.attribute_dims[0])
        for j in range(1, len(self.attribute_dims)):
            dims_temp.append(dims_temp[j - 1] + self.attribute_dims[j])
        dims_temp.insert(0, 0)
        dims_range = [(dims_temp[i - 1], dims_temp[i]) for i in range(1, len(dims_temp))]

        graph_relation = np.array(graph_relation >= self.beta*self.num_cases, dtype='int32')

        onehot_features = self.flat_onehot_features
        eye = np.eye(self.max_len, dtype=int)
        for case_index in range(self.num_cases):  #生成图
            attr_graphs = []
            edge = []
            xs = []
            ##构造顶点信息
            for attr_index in range(self.num_attributes):
                xs.append(
                    torch.tensor(onehot_features[case_index, :, dims_range[attr_index][0]:dims_range[attr_index][1]]))

            if self.case_lens[case_index]>1:
                ##构造边信息
                node = self.features[0][case_index,:self.case_lens[case_index]]
                for activity_index in range(0, self.case_lens[case_index]):
                    out = np.argwhere( graph_relation[self.features[0][case_index,activity_index]] == 1).flatten()
                    a = set(node)
                    b = set(out)
                    if activity_index+1< self.case_lens[case_index]:
                        edge.append([activity_index, activity_index+1])  #保证trace中相连的activity一定有边。
                    for node_name in a.intersection(b):
                        for node_index in np.argwhere(node == node_name).flatten():
                            if  activity_index+1 != node_index:
                                edge.append([activity_index, node_index])  # 添加有向边
            edge_index = torch.tensor(edge, dtype=torch.long)
            self.node_xs.append(xs)
            self.edge_indexs.append(edge_index.T)

        self.node_dims = self.attribute_dims.copy()

    def _gen_trace_graphs_GAE(self):

        dims_temp = []
        dims_temp.append(self.attribute_dims[0])
        for j in range(1, len(self.attribute_dims)):
            dims_temp.append(dims_temp[j - 1] + self.attribute_dims[j])
        dims_temp.insert(0, 0)
        dims_range = [(dims_temp[i - 1], dims_temp[i]) for i in range(1, len(dims_temp))]

        onehot_features = self.flat_onehot_features
        for case_index in range(self.num_cases):  #生成图
            edge = []
            xs = []
            edge_attr = None
            for attr_index in range(self.num_attributes):
                xs.append(onehot_features[case_index, :self.case_lens[case_index],
                          dims_range[attr_index][0]:dims_range[attr_index][1]])
            if self.case_lens[case_index]>1:
                ##构造顶点,边信息
                node=[]
                activity_array=xs[0]
                index_helper=[]
                for activity_index in range(0, self.case_lens[case_index]):
                    if list(activity_array[activity_index]) not in node:
                        node.append(list(activity_array[activity_index] )) #节点加入
                        index_helper.append(len(node)-1)
                    else:
                        index_helper.append(node.index(list(activity_array[activity_index])))

                edge=[[index_helper[i],index_helper[i+1]] for i in range(len(index_helper)-1)]


                for attr_index in range(1, self.num_attributes):
                    if edge_attr is None:
                        edge_attr = np.array(xs[attr_index])
                    else:
                        edge_attr = np.concatenate((edge_attr, xs[attr_index]), 1)
                if edge_attr is not None:
                    edge_attr = torch.tensor(edge_attr[:-1], dtype=torch.float )
            else:
                node=xs[0]
            edge_index = torch.tensor(edge, dtype=torch.long)
            if edge_attr is None:
                self.trace_graphs_GAE.append(Data(torch.tensor(np.array(node), dtype=torch.float), edge_index=edge_index.T))
            else:
                self.trace_graphs_GAE.append(
                    Data(torch.tensor(np.array(node), dtype=torch.float), edge_index=edge_index.T, edge_attr=edge_attr))

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

    def _load_dataset_from_cache(self, file):
        with gzip.open(file, 'rb') as f:
            (self._features, self.classes, self.labels, self._case_lens, self._attribute_dims,
             self.encoders, self.attribute_types, self.attribute_keys) = pickle.load(f)

    def _cache_dataset(self, file):
        with gzip.open(file, 'wb') as f:
            pickle.dump((self._features, self.classes, self.labels, self._case_lens, self._attribute_dims,
                         self.encoders, self.attribute_types, self.attribute_keys), f)

    @property
    def weak_labels(self):
        z = np.zeros(self.num_cases)
        for i in self.labeled_indices:
            z[i] = 1
        return z

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

    def __len__(self):
        return self.num_cases

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

    def get_indices_for_type(self, t):
        if len(self.text_labels) > 0:
            return np.where(self.text_labels == t)[0]
        else:
            return range(int(self.num_cases))

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

    @property
    def _reverse_features(self):
        reverse_features = [np.copy(f) for f in self._features]
        for f in reverse_features:
            for _f, m in zip(f, self.mask):
                _f[~m] = _f[~m][::-1]
        return reverse_features

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
    def onehot_features(self):
        """
        Return one-hot encoding of integer encoded features, while numerical features are passed as they are.

        As `features` this will return one tensor for each attribute. Shape of tensor for each attribute will be
        (number_of_cases, max_case_length, attribute_dimension). The attribute dimension refers to the number of unique
        values of the respective attribute encountered in the event log.

        :return:
        """
        one_hot_features = [self._to_categorical(f)[:, :, 1:] if t == AttributeType.CATEGORICAL else np.expand_dims(f, axis=2)
                for f, t in zip(self._features, self.attribute_types)]
        
        # RCVDB: Tensor seems to be of shape (attribute_dimension, number_of_cases, max_case_length)
        # print('One-hot features shape: ', len(one_hot_features), len(one_hot_features[0]), len(one_hot_features[0][0]))

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
        flat_one_hot_features = np.concatenate(self.onehot_features, axis=2)
        # print('Flat-One-hot features shape: ', len(flat_one_hot_features), len(flat_one_hot_features[0]), len(flat_one_hot_features[0][0]))

        return flat_one_hot_features

    @staticmethod
    def remove_time_dimension(x):
        return x.reshape((x.shape[0], np.product(x.shape[1:])))

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
        labels_event_level = np.zeros((num_cases, num_events, num_perspectives), dtype=int)
        labels_case_level = np.zeros((num_cases, num_perspectives), dtype=int)
        labels_attr_level = []

        for case_index, case in enumerate(event_log):
            case:Case
            case_targets = np.zeros((num_events, num_attributes, num_perspectives), dtype=int) 
            for event_index, event in enumerate(case.events):
                event:Event
                if event.attributes['_label'] is not None and '_label' in event.attributes:
                    event_labels = event.attributes['_label']
                    # if case_index < 2:
                    #     print(f'{case_index} {event_index} {event_labels}')
                    for label in event_labels:
                        # Encode the label into the targets:
                        case_targets, perspective = label_to_targets(case_targets, event_index, label)

                        # Debug code to see if labels are set correctly TODO can remove if not nessesary anymore
                        # if case_index < 2:
                        #     print(f' Setting value at {case_index},{event_index},{perspective}')

                        # Encode the perspective into labels
                        if perspective is not None: # If perspective is none then no anomalies need to be registered
                            labels_case_level[case_index, perspective] = 1
                            labels_event_level[case_index, event_index, perspective] = 1

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

    @staticmethod
    def _from_event_log(event_log:EventLog, include_attributes=None):
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
            case_lens.append(case.num_events + 2)  # +2 for start and end events
            for event in [start_event] + case.events + [end_event]:
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
        encoders = {}
        for key, attribute_type in zip(feature_columns.keys(), attr_types):
            # Integer encode categorical data
            if attribute_type == AttributeType.CATEGORICAL:
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                feature_columns[key] = encoder.fit_transform(feature_columns[key]) + 1
                encoders[key] = encoder

            # Normalize numerical data
            elif attribute_type == AttributeType.NUMERICAL:
                f = np.asarray(feature_columns[key])

                # RCVDB: Implementing min-max scaling to avoid relying on a normality assumption as experiments have shown that a normal distribution cannot be assumed in most features
                # feature_columns[key] = (f - f.mean()) / f.std()  # 0 mean and 1 std normalization
                f_min = np.min(f)
                f_max = np.max(f)
                if f_max != f_min:
                    feature_columns[key] = (f - f_min) / (f_max - f_min)
                else:
                    feature_columns[key] = np.zeros_like(f)

        # Transform back into sequences
        case_lens = np.array(case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        features = [np.zeros((case_lens.shape[0], case_lens.max()),dtype='int32') for _ in range(len(feature_columns))]
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
        print(f'Attribute Types: {attr_types}')
        print(f'Encoders: {encoders}')


        return features, case_lens, attr_types, encoders

    def from_event_log(self, event_log):
        """
        Load event log file and set the basic fields of the `Dataset` class.

        :param event_log: event log name as string
        :return:
        """
        # Get features from event log
        self._features, self._case_lens, self.attribute_types, self.encoders = self._from_event_log(event_log)

        # Get targets and labels from event log
        self.attr_labels, self.event_labels, self.case_labels = self._get_classes_and_labels_from_event_log(event_log)

        # Attribute keys (names)
        self.attribute_keys = [a.replace(':', '_').replace(' ', '_') for a in self.event_log.event_attribute_keys]


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