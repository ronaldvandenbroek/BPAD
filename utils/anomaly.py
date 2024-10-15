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
from processmining.log import EventLog
from utils.enums import Class, Perspective

 # RCVDB: TODO Rework all the prettify label functions
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
    def targets(event_log:EventLog, targets, event_index, label):
        """Return targets for the anomaly."""
        return targets, Perspective.NORMAL

    @staticmethod
    def pretty_label(label):
        """Return a text version of the label."""
        return 'Normal'


class NoneAnomaly(Anomaly):
    """Return the case unaltered, i.e., normal."""

    def __init__(self):
        super(NoneAnomaly, self).__init__()
        self.name = 'Normal'


class ReworkAnomaly(Anomaly):
    """Insert 1 sequence of n events coming from the case later in the case."""

    def __init__(self):
        super(ReworkAnomaly, self).__init__()

    @staticmethod
    def targets(event_log:EventLog, targets, event_index, label):
        attribute_index = event_log.get_activity_name()
        targets[event_index, attribute_index, Perspective.ORDER] = 1

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
    def targets(event_log:EventLog, targets, event_index, label):
        # RCVDB: Length of the skip can be ignored as that will not be present in the existing data
        attribute_index = event_log.get_activity_name()
        targets[event_index, attribute_index, Perspective.ORDER] = 1

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
    def targets(event_log:EventLog, targets, event_index, label):
        # RCVDB: Only the events that are shifted are marked as anomalous, each event will have the anomaly marked seperately
        attribute_index = event_log.get_activity_name()
        targets[event_index, attribute_index, Perspective.ORDER] = 1

        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        a = label['attr']['shift_from'] + 1
        b = label['attr']['shift_to'] + 1
        size = label['attr']['size']
        return f'{name} {size} from {a} to {b}'


class EarlyAnomaly(Anomaly):
    """Shift 1 sequence of `n` events by a distance `d` to the left."""

    def __init__(self):
        super(EarlyAnomaly, self).__init__()

    @staticmethod
    def targets(event_log:EventLog, targets, event_index, label):
        # RCVDB: Only the events that are shifted are marked as anomalous, each event will have the anomaly marked seperately
        attribute_index = event_log.get_activity_name()
        targets[event_index, attribute_index, Perspective.ORDER] = 1

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
    def targets(event_log:EventLog, targets, event_index, label):
        anomalous_attributes = label['attr']['attribute']
        for anomalous_attribute in anomalous_attributes:
            attribute_index = event_log.get_attribute_index(anomalous_attribute)
            targets[event_index, attribute_index, Perspective.ATTRIBUTE] = 1

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
    def targets(event_log:EventLog, targets, event_index, label):
        attribute_index = event_log.get_activity_name()
        targets[event_index, attribute_index, Perspective.ORDER] = 1 #Class.INSERT
        # RCVDB: Insert should not as the attributes in the inserted event are not themselves anomalous by default
        # targets[event_index, 1:, Perspective.ATTRIBUTE] = 1 #Class.ATTRIBUTE     
           
        return targets, Perspective.ORDER

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        indices = label['attr']['indices']
        return f'{name} at {", ".join([str(i + 1) for i in indices])}'

# RCVDB: Implementing arrival-time anomaly
class ArrivalTimeAnomaly(Anomaly):
    def __init__(self):
        super(ArrivalTimeAnomaly, self).__init__()

    @staticmethod
    def targets(event_log:EventLog, targets, event_index, label):
        attribute_index = event_log.get_attribute_index('arrival_time')
        targets[event_index, attribute_index, Perspective.ARRIVAL_TIME] = 1
        return targets, Perspective.ARRIVAL_TIME

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'

# RCVDB: Implementing global workload anomaly
class GlobalWorkloadAnomaly(Anomaly):
    def __init__(self):
        super(GlobalWorkloadAnomaly, self).__init__()

    @staticmethod
    def targets(event_log, targets, event_index, label):
        workload_timesteps = label['attr']['timestep']
        for workload_timestep in workload_timesteps:
            attribute_index = event_log.get_attribute_index(f'global_workload_{workload_timestep}')
            targets[event_index, attribute_index, Perspective.WORKLOAD] = 1

        return targets, Perspective.WORKLOAD

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'
    
# RCVDB: Implementing local workload anomaly
class LocalWorkloadAnomaly(Anomaly):
    def __init__(self):
        super(LocalWorkloadAnomaly, self).__init__()

    @staticmethod
    def targets(event_log:EventLog, targets, event_index, label):
        workload_timesteps = label['attr']['timestep']
        for workload_timestep in workload_timesteps:
            attribute_index = event_log.get_attribute_index(f'local_workload_{workload_timestep}')
            targets[event_index, attribute_index, Perspective.WORKLOAD] = 1

        return targets, Perspective.WORKLOAD

    @staticmethod
    def pretty_label(label):
        name = label['anomaly']
        start = label['attr']['start'] + 1
        inserted = label['attr']['inserted']
        return f'{name} {", ".join([e["name"] for e in inserted])} at {start}'

    
ANOMALIES = dict((s[:-7], anomaly) for s, anomaly in inspect.getmembers(sys.modules[__name__], inspect.isclass))

# RCVDB: Reworked label_to_targets to enable multi-label anomaly detection
def label_to_targets(event_log, targets, event_index, label):
    # If label is normal, then can skip
    if label == 'normal':
        return targets, None
    else:
        # RCVDB: TODO Check if event_index + 1 is needed as this is done in every anomaly
        event_index = event_index + 1
        # print(ANOMALIES)
        # print(label['anomaly'])
        anomaly:Anomaly = ANOMALIES.get(label['anomaly'])
        return anomaly.targets(event_log, targets, event_index, label)

# RCVDB: TODO prettify label
def prettify_label(label):
    if label == 'normal':
        return 'Normal'
    else:
        return ANOMALIES.get(label['anomaly']).pretty_label(label)
