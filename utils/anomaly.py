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
 # RCVDB: TODO Merge the detection code with the generation code of the other version
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
        pass

    def path_to_case(self, p, label=None):
        """
        Converts a given path to a case by traversing the graph and returning a case.

        :param p: path of node identifiers
        :param label: is used to label the case
        :return: a case
        """
        pass


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

    def __init__(self):
        super(ReworkAnomaly, self).__init__()

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

    def __init__(self):
        super(SkipSequenceAnomaly, self).__init__()

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

    def __init__(self):
        super(LateAnomaly, self).__init__()

    @staticmethod
    def targets(targets, event_index, label):
        # RCVDB: TODO Check if shift_from and shift_to is needed
        targets[event_index, 0, Perspective.ORDER]

        # s = label['attr']['shift_from'] + 1
        # e = label['attr']['shift_to'] + 1
        # size = label['attr']['size']
        # targets[s, 0, Perspective.ORDER] = 1 #Class.SHIFT
        # targets[e:e + size, 0, Perspective.ORDER] = 1 #Class.LATE
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

    def __init__(self):
        super(EarlyAnomaly, self).__init__()

    @staticmethod
    def targets(targets, event_index, label):
        # RCVDB: TODO Check if shift_from and shift_to is needed
        targets[event_index, 0, Perspective.ORDER]

        # s = label['attr']['shift_from'] + 1
        # e = label['attr']['shift_to'] + 1
        # size = label['attr']['size']
        # targets[s, 0, Perspective.ORDER] = 1 #Class.SHIFT
        # targets[e:e + size, 0, Perspective.ORDER] = 1 #Class.LATE
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

    def __init__(self):
        super(AttributeAnomaly, self).__init__()

    @staticmethod
    def targets(targets, event_index, label):
        # print(label['attr'])
        # RCVDB: TODO Check if these attribute indexes are correct
        # RCVDB: TODO Attribute index does not seem to exist, current workaround is just to ignore the index
        # attribute_indices = label['attr']['index']
        attribute_indices = [0] * len(label['attr']['attribute'])
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

    def __init__(self):
        super(InsertAnomaly, self).__init__()

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

    def __init__(self):
        super(SkipAnomaly, self).__init__()

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

    def __init__(self):
        super(ArrivalTimeAnomaly, self).__init__()

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
class GlobalWorkloadAnomaly(Anomaly):
    """Change n event timestamps to be outside of the expected distribution"""

    def __init__(self):
        super(GlobalWorkloadAnomaly, self).__init__()

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
class LocalWorkloadAnomaly(Anomaly):
    """Change n event timestamps to be outside of the expected distribution"""

    def __init__(self):
        super(LocalWorkloadAnomaly, self).__init__()

    @staticmethod
    def targets(targets, event_index, label):
        # RCVDB: Differs from the global workload in that it targets the resource
        # RCVDB: TODO Ensure that the resource is always in the 1st attribute index
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
        # print(ANOMALIES)
        # print(label['anomaly'])
        anomaly:Anomaly = ANOMALIES.get(label['anomaly'])
        return anomaly.targets(targets, event_index, label)

# RCVDB: TODO prettify label
def prettify_label(label):
    if label == 'normal':
        return 'Normal'
    else:
        return ANOMALIES.get(label['anomaly']).pretty_label(label)
