from datetime import datetime
import itertools
import os
import traceback
import time
import warnings
# import mlflow
from multiprocessing import Process
import multiprocessing

import numpy as np
from experiments.report_experiments import Experiment_Anomaly_Percentage, Experiment_Batch_Size, Experiment_Finetuning_T2V_Window_Vector_Sizes, Experiment_Finetuning_W2V_Window_Vector_Sizes, Experiment_Prefix, Experiment_Finetuning_Fixed_Vector_Vector_Sizes
from utils.dataset import Dataset

from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, RESULTS_RAW_DIR, ROOT_DIR, FSSave

# RCVDB: Supressing Sklearn LabelEncoder InconsistentVersionWarning as this seems an internal package issue
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fit_and_eva(dataset_name, run_name, seed, ad, fit_kwargs=None, ad_kwargs=None):
    print(fit_kwargs, ad_kwargs)
    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    batch_size = fit_kwargs['batch_size']
    prefix = fit_kwargs['prefix']
    bucket_boundaries = fit_kwargs['bucket_boundaries']
    categorical_encoding = fit_kwargs['categorical_encoding']
    numerical_encoding = fit_kwargs['numerical_encoding']
    vector_size = fit_kwargs['vector_size']
    window_size = fit_kwargs['window_size']
    pretrain_percentage = fit_kwargs['pretrain_percentage']
    try:
        repeats = fit_kwargs['repeats']
    except:
        repeats = None

    np.random.seed(seed)

    start_time = time.time()

    # AD
    ad = ad(**ad_kwargs)
    print(ad.name, dataset_name)

    fs_save = FSSave(start_time=datetime.now(), 
                     run_name=run_name, 
                     model_name=ad.name, 
                     categorical_encoding=categorical_encoding, 
                     numerical_encoding=numerical_encoding)
    dataset = Dataset(dataset_name, 
                      beta=0.005, 
                      prefix=prefix,
                      pretrain_percentage=pretrain_percentage,
                      vector_size=vector_size,
                      window_size=window_size, 
                      categorical_encoding=categorical_encoding,
                      numerical_encoding=numerical_encoding,
                      fs_save=fs_save)
    
    # if bucket_boundaries is not None:
    #     step=1
    #     bucket_boundaries = list(range(3,dataset.max_len,step))
    #     bucket_boundaries.append(dataset.max_len)

    # Run the AD model
    (
        bucket_trace_level_abnormal_scores, 
        bucket_event_level_abnormal_scores, 
        bucket_attr_level_abnormal_scores, 
        bucket_losses, bucket_case_labels, 
        bucket_event_labels, 
        bucket_attr_labels,
        bucket_errors_raw, 
        attribute_perspectives,
        attribute_perspectives_original, 
        attribute_names, 
        attribute_names_original
     ) = ad.train_and_predict(
        dataset, 
        batch_size=batch_size, 
        bucket_boundaries=bucket_boundaries, 
        categorical_encoding=categorical_encoding,
        vector_size=vector_size
    )

    end_time = time.time()
    run_time=end_time-start_time
    print(f'Runtime: {run_time}')

    config = fit_kwargs
    config['model'] = ad.name
    config['dataset'] = dataset_name
    config['seed'] = seed
    config['run_name'] = run_name
    config['start_time'] = start_time
    config['end_time'] = end_time
    config['run_time'] = run_time
    config['repeat'] = repeats
    config['attribute_perspectives'] = list(attribute_perspectives)
    config['attribute_perspectives_original'] = list(attribute_perspectives_original)
    config['attribute_names'] = list(attribute_names)
    config['attribute_names_original'] = list(attribute_names_original)
    fs_save.save_config(config)

    # RCVDB: Loop through each bucket size and handle each size seperately
    for i in range(len(bucket_losses)):
        if bucket_boundaries is not None:
            fs_save.set_bucket_size(bucket_boundaries[i])

        trace_level_abnormal_scores = bucket_trace_level_abnormal_scores[i]
        event_level_abnormal_scores = bucket_event_level_abnormal_scores[i]
        attr_level_abnormal_scores = bucket_attr_level_abnormal_scores[i]
        case_labels = bucket_case_labels[i]
        event_labels = bucket_event_labels[i]
        attr_labels = bucket_attr_labels[i]
        losses = bucket_losses[i]
        errors_raw = bucket_errors_raw[i]

        # fs_save.save_raw_errors(
        #     errors=errors_raw)
        fs_save.save_raw_labels(
            level='trace', 
            labels=case_labels)
        fs_save.save_raw_labels(
            level='event', 
            labels=event_labels)
        fs_save.save_raw_labels(
            level='attribute', 
            labels=attr_labels)
        fs_save.save_raw_losses(
            losses=losses)

        # RCVDB: Loop through each perspective and handle each result level seperately
        for anomaly_perspective in trace_level_abnormal_scores.keys():
            fs_save.set_perspective(anomaly_perspective)

            fs_save.save_raw_results( 
                level='trace',
                results=trace_level_abnormal_scores[anomaly_perspective])
            fs_save.save_raw_results(
                level='event',
                results=event_level_abnormal_scores[anomaly_perspective])
            fs_save.save_raw_results(
                level='attribute',
                results=attr_level_abnormal_scores[anomaly_perspective])

def execute_runs(dataset_names, ads, run_name, seed):
    print(f'Starting run: {run_name}')
    print(f'Total Planned configurations: {len(ads)}')
    print(f'Total Number of datasets: {len(dataset_names)}')
    for ad in ads:
        for d in dataset_names:
            p = Process(target=fit_and_eva, kwargs={ 'dataset_name' : d,  'run_name' : run_name, 'seed': seed, **ad })
            p.start()
            p.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    dataset_names = os.listdir(EVENTLOG_DIR)
    dataset_names.sort()
    if 'cache' in dataset_names:
        dataset_names.remove('cache')

    dataset_names_syn = [name for name in dataset_names if (
                                                        'gigantic' in name
                                                        or 'huge' in name
                                                        or 'large' in name
                                                        or 'medium' in name
                                                        or 'p2p' in name
                                                        or 'paper' in name
                                                        or 'small' in name
                                                        or 'wide' in name
    )]

    dataset_names_real = list(set(dataset_names)-set(dataset_names_syn))
    dataset_names_real.sort()


    seed=2024

    # ads, run_name = Experiment_Batch_Size(repeats=1)

    # TODO Can be run later
    # ads, run_name = Experiment_Prefix(repeats=1)

    # ads, run_name = Experiment_Anomaly_Percentage(repeats=1)
    # run_name = 'Experiment_Anomaly_Percentage_v2'

    # ads, run_name = Experiment_Synthetic_Dataset(repeats=1)
    # run_name = 'Experiment_Synthetic_Dataset_v5'

    # Finetuning runs
    ads, run_name = Experiment_Finetuning_Fixed_Vector_Vector_Sizes(repeats=3)
    execute_runs(dataset_names, ads, run_name, seed)

    ads, run_name = Experiment_Finetuning_W2V_Window_Vector_Sizes(repeats=3)
    execute_runs(dataset_names, ads, run_name, seed)

    ads, run_name = Experiment_Finetuning_T2V_Window_Vector_Sizes(repeats=3)
    execute_runs(dataset_names, ads, run_name, seed)