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

def fit_and_eva(dataset_name, run_name, ad, fit_kwargs=None, ad_kwargs=None):
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
    w2v_vector_size = fit_kwargs['w2v_vector_size']
    w2v_window_size = fit_kwargs['w2v_window_size']
    pretrain_percentage = fit_kwargs['pretrain_percentage']

    start_time = time.time()


    # if bucket_boundaries is not None:
    #     bucket_boundaries.append(dataset.max_len)

    # AD
    ad = ad(**ad_kwargs)
    print(ad.name, dataset_name)

    fs_save = FSSave(start_time=datetime.now(), run_name=run_name, model_name=ad.name)
    dataset = Dataset(dataset_name, 
                      beta=0.005, 
                      prefix=prefix,
                      pretrain_percentage=pretrain_percentage,
                      w2v_vector_size=w2v_vector_size,
                      w2v_window_size=w2v_window_size, 
                      categorical_encoding=categorical_encoding,
                      numerical_encoding=numerical_encoding,
                      fs_save=fs_save)

    bucket_trace_level_abnormal_scores, bucket_event_level_abnormal_scores, bucket_attr_level_abnormal_scores, bucket_losses, bucket_case_labels, bucket_event_labels, bucket_attr_labels = ad.train_and_predict(dataset, 
                                                                                                                                                                                                                 batch_size=batch_size, 
                                                                                                                                                                                                                 bucket_boundaries=bucket_boundaries, 
                                                                                                                                                                                                                 categorical_encoding=categorical_encoding,
                                                                                                                                                                                                                 w2v_vector_size=w2v_vector_size)

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


    run_name = 'W2V_Gridsearch_v3_batch_8_no_pretrain'
    w2v_vector_sizes = [20,40,60,80,100,150,200]
    w2v_window_size = [2,4,6,8,10]
    # pre_train_percentage = [0,0.1,0.5]
    pre_train_percentage = [0]

    ads_w2v_embedding_search = []
    w2v_combinations = list(itertools.product(w2v_vector_sizes, w2v_window_size, pre_train_percentage))
    for w2v_combination in w2v_combinations:
        w2v_vector_size, w2v_window_size, pre_train_percentage = w2v_combination

        ads_w2v_embedding_search.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=8, 
            prefix=True, 
            bucket_boundaries = [3,4,5,6,7,8,9],
            categorical_encoding=EncodingCategorical.WORD_2_VEC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=pre_train_percentage,
            w2v_vector_size = w2v_vector_size,
            w2v_window_size = w2v_window_size)))
        
    ads_w2v_embedding_search.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=8, 
            prefix=True, 
            bucket_boundaries = [3,4,5,6,7,8,9],
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            w2v_vector_size=0,
            w2v_window_size=0)))

    # RCVDB: Configuration to test multi-label anomalies
    # In practice probably more accurrate to have one epoch and a batch size of one to simulate each event arriving seperately
    # ads = [
    #     dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=8, 
    #         prefix=True, 
    #         bucket_boundaries = [3,4,5,6,7,8,9],
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0.5,
    #         w2v_vector_size = 100,
    #         w2v_window_size = 10)),
        # dict(ad=DAE, fit_kwargs=dict( 
        #     batch_size=1, 
        #     prefix=True, 
        #     bucket_boundaries = None, # [3,4,5,6,7,8,9], #None
        #     categorical_encoding=EncodingCategorical.WORD_2_VEC,
        #     numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
        #     w2v_vector_size = 100,
        #     w2v_window_size = 10)),
        # dict(ad=DAE, fit_kwargs=dict(
        #     batch_size=1, 
        #     prefix=True, 
        #     bucket_boundaries = None,
        #     categorical_encoding=EncodingCategorical.ONE_HOT,
        #     numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
        #     w2v_vector_size = 100,
        #     w2v_window_size = 10)),
        # dict(ad=DAE, fit_kwargs=dict(
        #     batch_size=1, 
        #     prefix=True, 
        #     bucket_boundaries = [5,8], # [3,4,5,6,7,8,9], #None
        #     categorical_encoding=EncodingCategorical.WORD_2_VEC,
        #     numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
        #     w2v_vector_size = 100,
        #     w2v_window_size = 10)),
        # dict(ad=DAE, fit_kwargs=dict(
        #     batch_size=1, 
        #     prefix=True, 
        #     bucket_boundaries = [5,8],
        #     categorical_encoding=EncodingCategorical.ONE_HOT,
        #     numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
        #     w2v_vector_size = 100,
        #     w2v_window_size = 10)),
        # dict(ad=DAE, fit_kwargs=dict( 
        #     batch_size=8, 
        #     prefix=True, 
        #     bucket_boundaries = [3,4,5,6,7,8,9], #None
        #     categorical_encoding=EncodingCategorical.WORD_2_VEC,
        #     numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
        #     w2v_vector_size = 100,
        #     w2v_window_size = 10)),
        # dict(ad=DAE, fit_kwargs=dict( 
        #     batch_size=8, 
        #     prefix=True, 
        #     bucket_boundaries = [3,4,5,6,7,8,9],
        #     categorical_encoding=EncodingCategorical.ONE_HOT,
        #     numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
        #     w2v_vector_size = 100,
        #     w2v_window_size = 10)),
    # ]

    # RCVDB: Full Configuration
    # ads = [
    #     dict(ad=LikelihoodPlusAnomalyDetector),  ## Multi-perspective, attr-level    --- Multi-perspective anomaly detection in business process execution events (extended to support the use of external threshold)
    #     dict(ad=NaiveAnomalyDetector),  # Control flow, trace-level    ---Algorithms for anomaly detection of traces in logs of process aware information systems
    #     dict(ad=SamplingAnomalyDetector),  # Control flow, trace-level    ---Algorithms for anomaly detection of traces in logs of process aware information systems
    #     dict(ad=DAE, fit_kwargs=dict(epochs=100, batch_size=64)),  ## Multi-perspective, attr-level    ---Analyzing business process anomalies using autoencoders
    #     dict(ad=BINetv3, fit_kwargs=dict(epochs=20, batch_size=64)), ## Multi-perspective, attr-level  ---BINet: Multi-perspective business process anomaly classification
    #     dict(ad=BINetv2, fit_kwargs=dict(epochs=20, batch_size=64)), ## Multi-perspective, attr-level  ---BINet: Multivariate business process anomaly detection using deep learning
    #     dict(ad=GAMA,ad_kwargs=dict(n_epochs=20)), ## Multi-perspective, attr-level    ---GAMA: A Multi-graph-based Anomaly Detection Framework for Business Processes via Graph Neural Networks
    #     dict(ad=VAE), ## Multi-perspective, attr-level 自己修改后使其能够检测attr-level      ---Autoencoders for improving quality of process event logs
    #     dict(ad=LAE), ## Multi-perspective, attr-level  自己修改后使其能够检测attr-level      ---Autoencoders for improving quality of process event logs
    #     dict(ad=GAE), ## Multi-perspective, trace-level       ---Graph Autoencoders for Business Process Anomaly Detection
    #     dict(ad=GRASPED), ## Multi-perspective, attr-level    ---GRASPED: A GRU-AE Network Based Multi-Perspective Business Process Anomaly Detection Model
    #     dict(ad=Leverage), # Control flow, trace-level       ---Keeping our rivers clean: Information-theoretic online anomaly detection for streaming business process events
    #     dict(ad=W2VLOF), # Control flow, trace-level     ---Anomaly Detection on Event Logs with a Scarcity of Labels
    #     dict(ad=VAEOCSVM) # Control flow, trace-level   ---Variational Autoencoder for Anomaly Detection in Event Data in Online Process Mining
    # ]

    print(f'Total Planned configurations: {len(ads_w2v_embedding_search)}')
    print(f'Total Number of datasets: {len(dataset_names)}')
    for ad in ads_w2v_embedding_search:
        for d in dataset_names:
            p = Process(target=fit_and_eva, kwargs={ 'dataset_name' : d,  'run_name' : run_name, **ad })
            p.start()
            p.join()