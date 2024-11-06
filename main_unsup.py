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
import pandas as pd

from baseline.GAE.gae import GAE
from baseline.GAMA.gama import GAMA
from baseline.GRASPED.grasped import GRASPED
from baseline.LAE.lae import LAE
from baseline.Sylvio import W2VLOF
from baseline.VAE.vae import VAE
from baseline.VAEOCSVM.vaeOCSVM import VAEOCSVM
from experiments.dea_experiments import DAE_bpic2015, DAE_bpic2015_no_buckets, DAE_finetuned_embedding, DAE_gridsearch_batch_bucketing, DAE_repeatability_experiment
from experiments.elmo_experiments import ELMo_finetuned
from experiments.fixed_vector_experiments import Fixed_Vector_gridsearch_vector_sizes
from experiments.general_experiments import All_original_models_finetuned
from experiments.t2v_experiments import T2V_finetuned
from experiments.w2v_experiments import W2V_finetuned, W2V_gridsearch_vector_window_size, W2V_no_averaging, W2V_pretrain
from novel.dae.dae import DAE
from baseline.bezerra import SamplingAnomalyDetector, NaiveAnomalyDetector
from baseline.binet.binet import BINetv3, BINetv2
from baseline.boehmer import LikelihoodPlusAnomalyDetector
from baseline.leverage import Leverage
from utils.dataset import Dataset

from utils.enums import Perspective, EncodingCategorical, EncodingNumerical
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

    np.random.seed(seed)

    start_time = time.time()


    # if bucket_boundaries is not None:
    #     bucket_boundaries.append(dataset.max_len)

    # AD
    ad = ad(**ad_kwargs)
    print(ad.name, dataset_name)

    fs_save = FSSave(start_time=datetime.now(), run_name=run_name, model_name=ad.name, config=fit_kwargs, categorical_encoding=categorical_encoding, numerical_encoding=numerical_encoding)
    dataset = Dataset(dataset_name, 
                      beta=0.005, 
                      prefix=prefix,
                      pretrain_percentage=pretrain_percentage,
                      vector_size=vector_size,
                      window_size=window_size, 
                      categorical_encoding=categorical_encoding,
                      numerical_encoding=numerical_encoding,
                      fs_save=fs_save)

    bucket_trace_level_abnormal_scores, bucket_event_level_abnormal_scores, bucket_attr_level_abnormal_scores, bucket_losses, bucket_case_labels, bucket_event_labels, bucket_attr_labels = ad.train_and_predict(dataset, 
                                                                                                                                                                                                                 batch_size=batch_size, 
                                                                                                                                                                                                                 bucket_boundaries=bucket_boundaries, 
                                                                                                                                                                                                                 categorical_encoding=categorical_encoding,
                                                                                                                                                                                                                 vector_size=vector_size)

    end_time = time.time()
    run_time=end_time-start_time
    print(f'Runtime: {run_time}')

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

        # RCVDB: Loop through each perspective and handle each results seperately
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
    # ads,run_name = W2V_pretrain()
    # ads,run_name = DAE_repeatability_experiment()
    # ads,run_name = W2V_gridsearch_vector_window_size()
    # ads,run_name = W2V_no_averaging()
    # ads,run_name = Fixed_Vector_gridsearch_vector_sizes()
    # ads,run_name = W2V_finetuned()
    # ads,run_name = DAE_gridsearch_batch_bucketing()
    # ads,run_name = All_original_models_finetuned()
    ads,run_name = DAE_bpic2015(
                        run_name='DAE_bpic2015_prefixes',
                        batch_size=8,
                        bucket=[20,30,40,50,60],
                        repeats=2,
                        prefix=True)

    # ads_small,run_name = T2V_finetuned(
    #                     run_name='Trace2Vec_Synthetic',
    #                     vector_sizes=[20],
    #                     window_sizes=[2],
    #                     batch_sizes=[8],
    #                     buckets=[[3,4,5,6,7,8,9]],
    #                     repeats=3,
    #                     prefix=True)
    # ads_large,run_name = T2V_finetuned(
    #                     run_name='Trace2Vec_Synthetic',
    #                     vector_sizes=[200],
    #                     window_sizes=[10],
    #                     batch_sizes=[8],
    #                     buckets=[[3,4,5,6,7,8,9]],
    #                     repeats=3,
    #                     prefix=True)
    # ads = ads_small + ads_large

    # ads, run_name = ELMo_finetuned(
    #                     run_name='ELMo_Synthetic',
    #                     vector_size=1024,
    #                     batch_size=8,
    #                     bucket=[3,4,5,6,7,8,9],
    #                     repeats=1,
    #                     prefix=True)
    

    # ads, run_name = DAE_bpic2015_no_buckets(
    #                     run_name='DAE_bpic2015_no_buckets_real_world',
    #                     batch_size=8,
    #                     bucket=None,
    #                     repeats=1,
    #                     prefix=True)

    # ads,run_name = DAE_finetuned_embedding()
    # run_name = "Mem Test"

    print(f'Total Planned configurations: {len(ads)}')
    print(f'Total Number of datasets: {len(dataset_names)}')
    for ad in ads:
        for d in dataset_names:
            p = Process(target=fit_and_eva, kwargs={ 'dataset_name' : d,  'run_name' : run_name, 'seed': seed, **ad })
            p.start()
            p.join()