import json
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from utils.fs import RESULTS_RAW_DIR


def list_subfolders_or_zip_files(experiment_name):
    experiment_path = os.path.join(RESULTS_RAW_DIR, experiment_name)
    run_names = [name for name in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, name)) or (name.endswith('.zip') and os.path.isfile(os.path.join(experiment_path, name)))]
    return run_names

def get_temp_folder_path(directory, run_name):
    return os.path.join(RESULTS_RAW_DIR, directory, run_name)

def unzip_results(directory, run_name):
    if run_name.endswith(".zip"):
        zip_path = os.path.join(RESULTS_RAW_DIR, directory, run_name)
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                new_run_name = os.path.splitext(run_name)[0]
                zip_ref.extractall(get_temp_folder_path(directory, new_run_name))
            return new_run_name, True
    else:
        return run_name, False
    
def cleanup_temp_folders(directory, run_name):
    temp_path = get_temp_folder_path(directory, run_name)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

def load_results(run_name, verbose=False, directory=None):
    if directory:
        run_path = os.path.join(RESULTS_RAW_DIR, directory, run_name)
    else:
        run_path = os.path.join(RESULTS_RAW_DIR, run_name)

    np_files = [file for file in os.listdir(run_path) if file.endswith('.npy') or file.endswith('.npz')]
    # print(np_files)

    loaded_data = {}

    # Load each .npy or .npz file and use the file name (without extension) as the key
    for np_file in np_files:
        file_path = os.path.join(run_path, np_file)
        key = os.path.splitext(np_file)[0]  # Get the file name without .npy or .npz extension

        if np_file.endswith('.npy'):
            # Directly load .npy files
            loaded_data[key] = np.load(file_path)
        elif np_file.endswith('.npz'):
            # Safely load .npz files and close the file afterward
            with np.load(file_path) as data:
                if len(data.files) == 1:
                    try:
                        loaded_data[key] = data[data.files[0]]  # Extract the single array
                    except Exception as e:
                        print(f"Error loading file: {np_file}: {e}")
                else:
                    raise ValueError(f"Multiple arrays in .npz file: {file_path}. Expected only one.")

        if verbose: print(f"{loaded_data[key]} \t {key}")

    return loaded_data

def load_config(run_name, directory=None):
    if directory:
        config_path = os.path.join(RESULTS_RAW_DIR, directory, run_name, "config.json")
    else:
        config_path = os.path.join(RESULTS_RAW_DIR, run_name, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def get_buckets(keys):
    buckets = set()
    for key in keys:
        numbers = re.findall(r'\d+', key)
        buckets.update(map(int, numbers))
    if len(buckets) > 0:
        return sorted(buckets)
    return None

def load_score_dataframe(file_path):
    if os.path.exists(file_path):
        loaded_df = pd.read_pickle(file_path)
        print("DataFrame loaded successfully!")
        return loaded_df
    else:
        print(f"The file {file_path} does not exist.")
        return None