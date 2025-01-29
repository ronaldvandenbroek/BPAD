#  Copyright 2018 Timo Nolle
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  ==============================================================================
from enum import Enum


class AttributeType(Enum):
    CATEGORICAL = 0
    NUMERICAL = 1

    @staticmethod
    def values():
        return ['Categorical', 'Numerical']

    @staticmethod
    def keys():
        return [AttributeType.CATEGORICAL, AttributeType.NUMERICAL]

    @staticmethod
    def items():
        return dict(zip(AttributeType.keys(), AttributeType.values()))


class   Axis(object):
    CASE = 0
    EVENT = 1
    ATTRIBUTE = 2

    @staticmethod
    def values():
        return ['Case', 'Event', 'Attribute']

    @staticmethod
    def keys():
        return [Axis.CASE, Axis.EVENT, Axis.ATTRIBUTE]

    @staticmethod
    def items():
        return dict(zip(Axis.keys(), Axis.values()))


class Class(object):
    NORMAL_ATTRIBUTE = -1
    NORMAL = 0
    ANOMALY = 1
    INSERT = 2
    SKIP = 3
    REWORK = 4
    EARLY = 5
    LATE = 6
    SHIFT = 7
    REPLACE = 8
    ATTRIBUTE = 9

    @staticmethod
    def values():
        return ['Normal Attribute', 'Normal', 'Anomaly', 'Insert', 'Skip', 'Rework', 'Early', 'Late', 'Shift',
                'Replace', 'Attribute']

    @staticmethod
    def colors():
        return ['#F5F5F5', '#F5F5F5', '#F44336', '#3F51B5', '#F57F17', '#388E3C', '#f06292', '#c2185b', '#795548',
                '#AB47BC', '#ab47bc']

    @staticmethod
    def color(key):
        return dict(zip(Class.keys(), Class.colors())).get(key)

    @staticmethod
    def keys():
        return [Class.NORMAL_ATTRIBUTE, Class.NORMAL, Class.ANOMALY, Class.INSERT, Class.SKIP, Class.REWORK,
                Class.EARLY, Class.LATE, Class.SHIFT, Class.REPLACE, Class.ATTRIBUTE, Class.ATTRIBUTE]

    @staticmethod
    def items():
        return dict(zip(Class.keys(), Class.values()))

# RCVDB: Enum representing the possible anomaly perspectives
class Perspective(object):
    # RCVDB: Removing normal as that isn't a perspective that needs to be predicted,
    # All values being none indicateds a normal trace.
    # NORMAL = 0
    ORDER = 0
    ATTRIBUTE = 1
    ARRIVAL_TIME = 2
    WORKLOAD = 3

    @staticmethod
    def values():
        return ['Order', 'Attribute', 'Arrival Time', 'Workload'] #'Normal', 

    @staticmethod
    def colors():
        return ['#F44336', '#3F51B5', '#F57F17', '#388E3C'] #'#F5F5F5', 

    @staticmethod
    def color(key):
        return dict(zip(Perspective.keys(), Perspective.colors())).get(key)

    @staticmethod
    def keys():
        return [Perspective.ORDER, Perspective.ATTRIBUTE, Perspective.ARRIVAL_TIME, Perspective.WORKLOAD] #Perspective.NORMAL,

    @staticmethod
    def items():
        return dict(zip(Perspective.keys(), Perspective.values()))

class PadMode(object):
    PRE = 'pre'
    POST = 'post'

    @staticmethod
    def keys():
        return [PadMode.PRE, PadMode.POST]


class Mode(object):
    BINARIZE = 'binarize'
    CLASSIFY = 'classify'

    @staticmethod
    def values():
        return ['Binarize', 'Classify']

    @staticmethod
    def keys():
        return [Mode.BINARIZE, Mode.CLASSIFY]

    @staticmethod
    def items():
        return dict(zip(Mode.keys(), Mode.values()))


class Base(object):
    LEGACY = 'legacy'
    SCORES = 'scores'

    @staticmethod
    def values():
        return ['Legacy', 'Scores']

    @staticmethod
    def keys():
        return [Base.LEGACY, Base.SCORES]

    @staticmethod
    def items():
        return dict(zip(Base.keys(), Base.values()))


class Normalization(object):
    MINMAX = 'minmax'

    @staticmethod
    def values():
        return ['MinMax']

    @staticmethod
    def keys():
        return [Normalization.MINMAX]

    @staticmethod
    def items():
        return dict(zip(Normalization.keys(), Normalization.values()))


class Heuristic(object):
    DEFAULT = 'default'
    MANUAL = 'manual'
    BEST = 'best'
    ELBOW_DOWN = 'elbow'
    ELBOW_UP = 'broken_elbow'
    LP_LEFT = 'stable_left'
    LP_MEAN = 'stable_mean'
    LP_RIGHT = 'stable_right'
    MEAN = 'mean'
    MEDIAN = 'median'
    RATIO = 'ratio'

    @staticmethod
    def values():
        return [r'$default$', r'$manual$', r'$best$', r'$elbow_\downarrow$', r'$elbow_\uparrow$',
                r'$lp_\leftarrow$', r'$lp_\leftrightarrow$', r'$lp_\rightarrow$', r'$\bar{S}$', r'$\tilde{S}$',
                r'$ratio$']

    @staticmethod
    def keys():
        return [Heuristic.DEFAULT, Heuristic.MANUAL, Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO]

    @staticmethod
    def items():
        return dict(zip(Heuristic.keys(), Heuristic.values()))


class Strategy(object):
    DEFAULT = 'default'
    SINGLE = 'single'
    ATTRIBUTE = 'attribute'
    POSITION = 'position'
    POSITION_ATTRIBUTE = 'position_attribute'

    @staticmethod
    def values():
        return ['Default', r'$h$', r'$h^{(a)}$', r'$h^{(e)}$', r'$h^{(ea)}$']

    @staticmethod
    def keys():
        return [Strategy.DEFAULT, Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]

    @staticmethod
    def items():
        return dict(zip(Strategy.keys(), Strategy.values()))


class EncodingCategorical(Enum):
    NONE = 'None'
    ONE_HOT = 'One Hot'
    EMBEDDING = 'Embedding' #Not Implemented
    WORD_2_VEC_ATC = 'Word2Vec Average Then Concatinate'
    WORD_2_VEC_C = 'Word2Vec Concatinate'
    FIXED_VECTOR = 'Fixed Vector'
    TRACE_2_VEC_ATC = 'Trace2Vec Average Then Concatinate'
    TRACE_2_VEC_C = 'Trace2Vec Concatinate'
    ELMO = 'ELMo' #Not Implemented
    TOKENIZER = 'Tokenizer'

    @staticmethod
    def values():
        return ['None', 
                'One Hot', 
                'Embedding', 
                'Word2Vec Average Then Concatinate', 
                'Word2Vec Concatinate', 
                'Fixed Vector',
                'Trace2Vec Average Then Concatinate',
                'Trace2Vec Concatinate',
                'ELMo',
                'Tokenizer']
    
    @staticmethod
    def values_short():
        return ['None', 
                'One-Hot', 
                'Embedding', 
                'W2V-ATC', 
                'W2V-C', 
                'Fixed-Vector',
                'T2V-ATC',
                'T2V-C',
                'ELMo',
                'Tokenizer']

    @staticmethod
    def keys():
        return [EncodingCategorical.NONE, 
                EncodingCategorical.ONE_HOT, 
                EncodingCategorical.EMBEDDING, 
                EncodingCategorical.WORD_2_VEC_ATC,
                EncodingCategorical.WORD_2_VEC_C, 
                EncodingCategorical.FIXED_VECTOR,
                EncodingCategorical.TRACE_2_VEC_ATC,
                EncodingCategorical.TRACE_2_VEC_C,
                EncodingCategorical.ELMO,
                EncodingCategorical.TOKENIZER]

    @staticmethod
    def items():
        return dict(zip(EncodingCategorical.keys(), EncodingCategorical.values()))
    
    @staticmethod
    def items_short():
        return dict(zip(EncodingCategorical.keys(), EncodingCategorical.values_short()))
    
    @staticmethod
    def from_string(value):
        # Create a dictionary mapping each string value to the Enum member
        lookup = {v.value: v for v in EncodingCategorical}
        # Use the dictionary to return the matching enum member or raise KeyError
        return lookup.get(value)
    

class EncodingNumerical(Enum):
    NONE = 'None'
    MIN_MAX_SCALING = 'Min Max Scaling'

    @staticmethod
    def values():
        return ['None', 'Min Max Scaling']

    @staticmethod
    def keys():
        return [EncodingNumerical.NONE, EncodingNumerical.MIN_MAX_SCALING]

    @staticmethod
    def items():
        return dict(zip(EncodingNumerical.keys(), EncodingNumerical.values()))