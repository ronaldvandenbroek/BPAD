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
import os.path
from pathlib import Path

import numpy as np

from utils.enums import Perspective



# Base
ROOT_DIR = Path(__file__).parent.parent

# Output
EVENTLOG_DIR = os.path.join(ROOT_DIR,'eventlogs')  # For generated event logs
RESULTS_DIR = os.path.join(ROOT_DIR,'results')  # For the model scores
RESULTS_RAW_DIR = os.path.join(ROOT_DIR,'results','raw') # For the model raw error scores


# Cache
EVENTLOG_CACHE_DIR = os.path.join(ROOT_DIR,'eventlogs', 'cache')  # For caching datasets so the event log does not always have to be loaded

def save_raw_losses(start_time, model_name, losses):
    if not os.path.exists(RESULTS_RAW_DIR):
        os.makedirs(RESULTS_RAW_DIR)

    raw_results_path = os.path.join(RESULTS_RAW_DIR, f'result_{round(start_time)}_{model_name}_losses')
    np.save(raw_results_path, losses)  

def save_raw_results(start_time, model_name, level, perspective, results):
    if not os.path.exists(RESULTS_RAW_DIR):
        os.makedirs(RESULTS_RAW_DIR)

    perspective_name = Perspective.values()[perspective]
    raw_results_path = os.path.join(RESULTS_RAW_DIR, f'result_{round(start_time)}_{model_name}_{perspective_name}_{level}')
    np.save(raw_results_path, results)    

def split_eventlog_name(name):
    try:
        s = name.split('-')
        model = s[0]
        p = float(s[1])
        id = int(s[2])
    except Exception:
        model = None
        p = None
        id = None
    return model, p, id



class File(object):
    ext = None

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.file = self.path.name
        self.name = self.path.stem
        self.str_path = str(path)

    def remove(self):
        import os
        if self.path.exists():
            os.remove(self.path)


class EventLogFile(File):
    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        if '.json' not in path.suffixes:
            path = Path(str(path) + '.json.gz')
        if not path.is_absolute():
            path = Path(os.path.join(EVENTLOG_DIR , path.name))

        super(EventLogFile, self).__init__(path)

        if len(self.path.suffixes) > 1:
            self.name = Path(self.path.stem).stem

        self.model, self.p, self.id = split_eventlog_name(self.name)

    @property
    def cache_file(self):
        if not os.path.exists(EVENTLOG_CACHE_DIR):
            os.makedirs(EVENTLOG_CACHE_DIR)
        return Path(os.path.join(EVENTLOG_CACHE_DIR , (self.name + '.pkl.gz')))

