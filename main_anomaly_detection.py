from datetime import datetime
import os
import time
import numpy as np
from multiprocessing import Process
import multiprocessing

from utils.dataset import Dataset
from utils.fs import EVENTLOG_DIR, FSSave

def fit_and_eva(dataset_name, dataset_folder, run_name, seed, ad, fit_kwargs=None):
    print(fit_kwargs)
    if fit_kwargs is None:
        fit_kwargs = {}
    categorical_encoding = fit_kwargs['categorical_encoding']
    numerical_encoding = fit_kwargs['numerical_encoding']
 
    np.random.seed(seed)

    start_time = time.time()

    # AD
    ad = ad(fit_kwargs)
    print(ad.name, dataset_name)

    fs_save = FSSave(start_time=datetime.now(), 
                     run_name=run_name, 
                     model_name=ad.name, 
                     categorical_encoding=categorical_encoding, 
                     numerical_encoding=numerical_encoding)
    dataset = Dataset(dataset_name,
                      dataset_folder, 
                      beta=0.005, 
                      prefix=fit_kwargs.get('prefix', True),
                      pretrain_percentage=fit_kwargs.get('pretrain_percentage', 0),
                      vector_size=fit_kwargs.get('vector_size', 50),
                      window_size=fit_kwargs.get('window_size', 10),
                      categorical_encoding=categorical_encoding,
                      numerical_encoding=numerical_encoding,
                      fs_save=fs_save)
    
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
        attribute_names_original,
        processed_prefixes,
        processed_events,
        runtime_results
    ) = ad.train_and_predict(dataset)

    end_time = time.time()
    run_time=end_time-start_time
    print(f'Runtime: {run_time}')

    config = fit_kwargs
    config['model'] = fit_kwargs.get('model_name',  ad.name)
    config['dataset'] = dataset_name
    config['dataset_folder'] = dataset_folder
    config['seed'] = seed
    config['run_name'] = run_name
    config['start_time'] = start_time
    config['end_time'] = end_time
    config['run_time'] = run_time
    config['repeat'] = fit_kwargs.get('repeats', None)
    config['use_prefix_errors'] = fit_kwargs.get('use_prefix_errors', None)
    config['event_positional_encoding'] = fit_kwargs.get('event_positional_encoding', None)
    config['attribute_perspectives'] = list(attribute_perspectives)
    config['attribute_perspectives_original'] = list(attribute_perspectives_original)
    config['attribute_names'] = list(attribute_names)
    config['attribute_names_original'] = list(attribute_names_original)
    config['runtime_results'] = runtime_results
    # config['processed_prefixes'] = list(processed_prefixes)
    # config['processed_events'] = list(processed_events)
    fs_save.save_config(config)

    # RCVDB: Loop through each bucket size and handle each size seperately
    bucket_boundaries = fit_kwargs.get('bucket_boundaries', None)
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
    
    fs_save.zip_results()

def execute_runs(dataset_names, ads, run_name, dataset_folder, seed):
    multiprocessing.set_start_method('spawn')

    print(f'Starting run: {run_name}')
    print(f'Total Planned configurations: {len(ads)}')
    print(f'Total Number of datasets: {len(dataset_names)}')
    for ad in ads:
        for d in dataset_names:
            p = Process(target=fit_and_eva, kwargs={'dataset_name' : d, 'dataset_folder': dataset_folder, 'run_name' : run_name, 'seed': seed, **ad})
            p.start()
            p.join()

def prepare_datasets(dataset_folder=None):
    if dataset_folder:
        dataset_names = os.listdir(os.path.join(EVENTLOG_DIR,dataset_folder))
    else:
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

    return dataset_names