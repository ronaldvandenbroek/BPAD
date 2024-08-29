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

import inspect
import sys

import numpy as np

from processmining.case import Case
from processmining.event import Event
from utils.enums import Class, Perspective

 # RCVDB: TODO Rework all the prettify label functions
 # RCVDB: TODO Potentially delete all generation code from this class as that is handled by the other version of the code (or merge them)
class Anomaly(object):
    """Base class for anomaly implementations."""

    def __init__(self):
        self.graph = None
        self.activities = None
        self.attributes = None
        self.name = self.__class__.__name__[:-7]

    def __str__(self):
        return self.name

    @property
    def json(self):
        return dict(anomaly=str(self),
                    parameters=dict((k, v) for k, v in vars(self).items() if k not in ['graph', 'attributes']))

    @property
    def event_len(self):
        n = 1
        if self.attributes is not None:
            n += len(self.attributes)
        return n

    @staticmethod
    def targets(targets, event_index, label):
        """Return targets for the anomaly."""
        return targets, Perspective.NORMAL

    @staticmethod
    def pretty_label(label):
        """Return a text version of the label."""
        return 'Normal'

    def apply_to_case(self, case):
        """
        This method applies the anomaly to a given case

        :param case: the input case
        :return: a new case after the anomaly has been applied
        """
        pass

    def apply_to_path(self, path):
        """
        This method applies the anomaly to a given path in the graph.

        Requires self.graph to be set.

        :param path: the path containing node identifiers for the graph
        :return: a new case after anomaly has been applied
        """
        return self.apply_to_case(self.path_to_case(path))

    def path_to_case(self, p, label=None):
        """
        Converts a given path to a case by traversing the graph and returning a case.

        :param p: path of node identifiers
        :param label: is used to label the case
        :return: a case
        """
        g = self.graph

        case = Case(label=label)
        for i in range(0, len(p), self.event_len):
            event = Event(name=g.nodes[p[i]]['value'])
            for j in range(1, self.event_len):
                att = g.nodes[p[i + j]]['name']
                value = g.nodes[p[i + j]]['value']
                event.attributes[att] = value
            case.add_event(event)

        return case

    def generate_random_event(self):
        if self.activities is None:
            raise RuntimeError('activities has not bee set.')

        event = Event(name=f'Random activity {np.random.randint(1, len(self.activities))}')
        if self.attributes is not None:
            event.attributes = dict(
                (a.name, f'Random {a.name} {np.random.randint(1, len(a.values))}') for a in self.attributes)
        return event


class NoneAnomaly(Anomaly):
    """Return the case unaltered, i.e., normal."""

    def __init__(self):
        super(NoneAnomaly, self).__init__()
        self.name = 'Normal'

    def apply_to_case(self, case):
        case.attributes['label'] = 'normal'
        return case


class ReworkAnomaly(Anomaly):
    """Insert 1 sequence of n events coming from the case later in the case."""

    def __init__(self, max_sequence_size=2, max_distance=0):
        self.max_sequence_size = max_sequence_size
        self.max_distance = max_distance
        super(ReworkAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 1:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(2, min(len(case), self.max_sequence_size) + 1)
        start = np.random.randint(0, len(case) - size + 1)
        distance = np.random.randint(0, min(len(case) - (start + size), self.max_distance) + 1)

        t = case.events
        dupe_sequence = [Event.clone(e) for e in t[start:start + size]]

        inserted = [e.json for e in dupe_sequence]

        anomalous_trace = t[:start + size + distance] + dupe_sequence + t[start + size + distance:]
        case.events = anomalous_trace

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                size=int(size),
                start=int(start + size + distance),
                inserted=inserted
            )
        )

        return case

    @staticmethod
    def targets(targets, event_index, label):
        size = label['attr']['size']
        targets[event_index:event_index + size, 0, Perspective.ORDER] = 1
        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'


class SkipSequenceAnomaly(Anomaly):
    """Skip 1 sequence of n events."""

    def __init__(self, max_sequence_size=1):
        self.max_sequence_size = max_sequence_size
        super(SkipSequenceAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(1, min(len(case) - 1, self.max_sequence_size) + 1)
        start = np.random.randint(0, len(case) - size)
        end = start + size

        t = case.events
        skipped = [s.json for s in t[start:end]]
        case.events = t[:start] + t[end:]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                size=int(size),
                start=int(start),
                skipped=skipped
            )
        )

        return case

    @staticmethod
    def targets(targets, event_index, label):
        targets[event_index, 0, Perspective.ORDER] = 1
        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        skipped = label['attr']['skipped']
        return f'{name} {", ".join([e["name"] for e in skipped])} at {start}'


class LateAnomaly(Anomaly):
    """Shift 1 sequence of `n` events by a distance `d` to the right."""

    def __init__(self, max_distance=1, max_sequence_size=1):
        self.max_distance = max_distance
        self.max_sequence_size = max_sequence_size
        super(LateAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(1, min(len(case) - 1, self.max_sequence_size + 1))
        distance = np.random.randint(1, min(len(case) - size, self.max_distance + 1))
        s = np.random.randint(0, len(case) - size - distance)
        i = s + distance

        t = case.events

        case.events = t[:s] + t[s + size:i + size] + t[s:s + size] + t[i + size:]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                shift_from=int(s),
                shift_to=int(i),
                size=int(size)
            )
        )

        return case

    @staticmethod
    def targets(targets, event_index, label):
        # RCVDB: TODO Check if shift_from and shift_to is needed
        s = label['attr']['shift_from'] + 1
        e = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        targets[s, 0, Perspective.ORDER] = 1 #Class.SHIFT
        targets[e:e + size, 0, Perspective.ORDER] = 1 #Class.LATE
        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        a = label['attr']['shift_from'] + 1
        b = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        return f'{name} {size} from {a} to {b}'


class EarlyAnomaly(Anomaly):
    """Shift 1 sequence of `n` events by a distance `d` to the right."""

    def __init__(self, max_distance=1, max_sequence_size=1):
        self.max_distance = max_distance
        self.max_sequence_size = max_sequence_size
        super(EarlyAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 2:
            return NoneAnomaly().apply_to_case(case)

        size = np.random.randint(1, min(len(case) - 1, self.max_sequence_size + 1))
        distance = np.random.randint(1, min(len(case) - size, self.max_distance + 1))
        s = np.random.randint(distance, len(case) - size)
        i = s - distance

        t = case.events

        case.events = t[:i] + t[s:s + size] + t[i:s] + t[s + size:]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                shift_from=int(s + size),
                shift_to=int(i),
                size=int(size)
            )
        )

        return case

    @staticmethod
    def targets(targets, event_index, label):
        # RCVDB: TODO Check if shift_from and shift_to is needed
        s = label['attr']['shift_from'] + 1
        e = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        targets[s, 0, Perspective.ORDER] = 1 #Class.SHIFT
        targets[e:e + size, 0, Perspective.ORDER] = 1 #Class.LATE
        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        a = label['attr']['shift_from'] + 1
        b = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        return f'{name} {size} from {a} to {b}'


class AttributeAnomaly(Anomaly):
    """Replace n attributes in m events with an incorrect value."""

    def __init__(self, max_events=1, max_attributes=1):
        super(AttributeAnomaly, self).__init__()
        self.max_events = max_events
        self.max_attributes = max_attributes

    def apply_to_case(self, case):
        n = np.random.randint(1, min(len(case), self.max_events) + 1)
        event_indices = sorted(np.random.choice(range(len(case)), n, replace=False))
        # print(n)
        # print(case)
        # print(event_indices)
        indices = []
        original_attribute_values = []
        affected_attribute_names = []
        for event_index in event_indices:
            # print(self.attributes)
            if min(len(self.attributes), self.max_attributes) + 1 <=1 : #无属性
                return NoneAnomaly().apply_to_case(case)
            m = np.random.randint(1, min(len(self.attributes), self.max_attributes) + 1)
            attribute_indices = sorted(np.random.choice(range(len(self.attributes)), m, replace=False))
            for attribute_index in attribute_indices:
                affected_attribute = self.attributes[attribute_index]
                original_attribute_value = case[event_index].attributes[affected_attribute.name]

                indices.append(int(event_index))
                original_attribute_values.append(original_attribute_value)
                affected_attribute_names.append(affected_attribute.name)

                # Set the new value
                case[event_index].attributes[affected_attribute.name] = affected_attribute.random_value()

        attribute_names = sorted([a.name for a in self.attributes])
        attribute_indices = [attribute_names.index(a) for a in affected_attribute_names]

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                index=indices,
                attribute_index=attribute_indices,
                attribute=affected_attribute_names,
                original=original_attribute_values
            )
        )

        return case

    def apply_to_path(self, path):
        case_len = int(len(path) / self.event_len)

        n = np.random.randint(1, min(case_len, self.max_events) + 1)
        idx = sorted(np.random.choice(range(case_len), n, replace=False))

        indices = []
        original_attribute_values = []
        affected_attribute_names = []
        attribute_domains = []
        for index in idx:
            m = np.random.randint(1, min(len(self.attributes), self.max_attributes) + 1)
            attribute_indices = sorted(np.random.choice(range(len(self.attributes)), m, replace=False))
            for attribute_index in attribute_indices:
                affected_attribute = self.attributes[attribute_index]

                predecessor = path[index * self.event_len + attribute_index]

                attribute_values = [self.graph.nodes[s]['value'] for s in self.graph.successors(predecessor)]
                attribute_domain = [x for x in affected_attribute.domain if x not in attribute_values]
                original_attribute_value = self.graph.nodes[path[index * self.event_len + attribute_index + 1]]['value']

                indices.append(int(index))
                original_attribute_values.append(original_attribute_value)
                affected_attribute_names.append(affected_attribute.name)
                attribute_domains.append(attribute_domain)

        attribute_names = sorted([a.name for a in self.attributes])
        attribute_indices = [attribute_names.index(a) for a in affected_attribute_names]

        label = dict(
            anomaly=str(self),
            attr=dict(
                index=indices,
                attribute_index=attribute_indices,
                attribute=affected_attribute_names,
                original=original_attribute_values
            )
        )

        case = self.path_to_case(path, label)
        for index, affected_attribute, attribute_domain in zip(indices, affected_attribute_names, attribute_domains):
            case[index].attributes[affected_attribute] = np.random.choice(attribute_domain)

        return case

    @staticmethod
    def targets(targets, event_index, label):
        # print(label['attr'])
        # RCVDB: Check if these attribute indexes are correct
        attribute_indices = label['attr']['index']
        for index in attribute_indices:
            # RCVDB: index + 1 goes out of bounds
            # RCVDB: TODO Problem seems to be on the data generation side, where the attributes are mixed in with other columns
            # RCVDB: Potential fix is to make sure that the attribute columns are at the leftmost side
            # RCVDB: Temp solution here: Ignore all out of bounds attibutes
            if index < targets.shape[1]:
                targets[event_index, index, Perspective.ATTRIBUTE] = 1
        return targets, Perspective.ATTRIBUTE

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        affected = label['attr']['attribute']
        index = [i + 1 for i in label['attr']['index']]
        original = label['attr']['original']
        return f'{name} {affected} at {index} was {original}'

class InsertAnomaly(Anomaly):
    """Add n random events."""

    def __init__(self, max_inserts=1):
        self.max_inserts = max_inserts
        super(InsertAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) < 2:
            return NoneAnomaly().apply_to_case(case)

        num_inserts = np.random.randint(1, min(int(len(case) / 2), self.max_inserts) + 1)
        insert_places = sorted(np.random.choice(range(len(case) - 1), num_inserts, replace=False))
        insert_places += np.arange(len(insert_places))

        t = case.events
        for place in insert_places:
            t = t[:place] + [self.generate_random_event()] + t[place:]
        case.events = t

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                indices=[int(i) for i in insert_places]
            )
        )

        return case

    @staticmethod
    def targets(targets, event_index, label):
        targets[event_index, 0, Perspective.ORDER] = 1 #Class.INSERT
        targets[event_index, 1:, Perspective.ATTRIBUTE] = 1 #Class.ATTRIBUTE        
        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        indices = label['attr']['indices']
        return f'{name} at {", ".join([str(i + 1) for i in indices])}'


class SkipAnomaly(Anomaly):
    """Skip n single events."""

    def __init__(self, max_skips=2):
        self.max_skips = max_skips
        super(SkipAnomaly, self).__init__()

    def apply_to_case(self, case):
        if len(case) <= 4:
            return NoneAnomaly().apply_to_case(case)

        num_skips = np.random.randint(2, min(int(len(case) / 2), self.max_skips) + 1)
        skip_places = sorted(np.random.choice(range(len(case) - 1), num_skips, replace=False))

        skipped = [case.events[i].json for i in skip_places]
        t = [e for i, e in enumerate(case.events) if i not in skip_places]

        case.events = t

        skip_places -= np.arange(len(skip_places))
        skip_places = list(set(skip_places))

        case.attributes['label'] = dict(
            anomaly=str(self),
            attr=dict(
                indices=[int(i) for i in skip_places],
                skipped=skipped
            )
        )

        return case

    @staticmethod
    def targets(targets, event_index, label):
        targets[event_index, 0, Perspective.ORDER] = 1 #Class.SKIP
        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        indices = ', '.join([str(i + 1) for i in label['attr']['indices']])
        skipped = ', '.join([e['name'] for e in label['attr']['skipped']])
        return f'{name} {skipped} at {indices}'

# RCVDB: Implementing arrival-time anomaly
class ArrivalTimeAnomaly(Anomaly):
    """Change n event timestamps to be outside of the expected distribution"""

    def __init__(self, max_events=1, max_scale=2):
        self.max_events = max_events
        self.max_scale = max_scale
        self.perspective = 'arrival_time' # RCVDB: Specify in which anomaly perspective category the anomaly falls
        super(ArrivalTimeAnomaly, self).__init__()

    def apply_to_case(self, case:Case):
        n = np.random.randint(1, min(len(case), self.max_events) + 1)
        event_indices = sorted(np.random.choice(range(len(case)), n, replace=False))

        # RCVDB: Only mark which events should be ArrivalTime anomalous, timestamps are added in postprocessing 
        for event_index in event_indices:
            case[event_index].set_anomaly_label(dict(
                anomaly=str(self)
            ))

        case.set_anomaly_label(dict(
            anomaly=str(self)
        ))

        return case

    @staticmethod
    def targets(targets, event_index, label):
        targets[event_index, 0, Perspective.ARRIVAL_TIME] = 1
        return targets, Perspective.ARRIVAL_TIME

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'

# RCVDB: Implementing global workload anomaly
class GlobalWorkLoadAnomaly(Anomaly):
    """Change n event timestamps to be outside of the expected distribution"""

    def __init__(self):
        super(GlobalWorkLoadAnomaly, self).__init__()

    @staticmethod
    def targets(targets, event_index, label):
        targets[event_index, 0, Perspective.WORKLOAD] = 1
        return targets, Perspective.WORKLOAD

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'
    
# RCVDB: Implementing local workload anomaly
class LocalWorkLoadAnomaly(Anomaly):
    """Change n event timestamps to be outside of the expected distribution"""

    def __init__(self):
        super(LocalWorkLoadAnomaly, self).__init__()

    @staticmethod
    def targets(targets, event_index, label):
        # RCVDB: Differs from the global workload in that it targets the resource
        # RCVDBL TODO Ensure that the resource is always in the 1st attribute index
        targets[event_index, 1, Perspective.WORKLOAD] = 1
        return targets, Perspective.WORKLOAD

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'

    
ANOMALIES = dict((s[:-7], anomaly) for s, anomaly in inspect.getmembers(sys.modules[__name__], inspect.isclass))

# RCVDB: Reworked label_to_targets to enable multi-label anomaly detection
def label_to_targets(targets, event_index, label):
    # If label is normal, then can skip
    if label == 'normal':
        return targets, Perspective.NORMAL
    else:
        # RCVDB: TODO Check if event_index + 1 is needed as this is done in every anomaly
        event_index = event_index + 1

        anomaly:Anomaly = ANOMALIES.get(label['anomaly'])
        return anomaly.targets(targets, event_index, label)

# RCVDB: TODO prettify label
def prettify_label(label):
    if label == 'normal':
        return 'Normal'
    else:
        return ANOMALIES.get(label['anomaly']).pretty_label(label)
