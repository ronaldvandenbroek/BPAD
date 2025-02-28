import itertools
import numpy as np
from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical


def Experiment_Real_World_W2V_ATC(repeats=5):
    bucket = [10,20,30,40,60,80]
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Real_World_W2V_ATC'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=2)))
    return ads, run_name

def Experiment_Real_World_T2V_ATC(repeats=5):
    bucket = [10,20,30,40,60,80]
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Real_World_T2V_ATC'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=4)))
    return ads, run_name

def Experiment_Real_World_T2V_C(repeats=5):
    bucket = [10,20,30,40,60,80]
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Real_World_T2V_C'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=2)))
    return ads, run_name
        

def Experiment_Synthetic_All_Models_FV_OH(repeats=5):
    bucket = [3,4,5,6,7,8,9]
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Synthetic_All_Models_FV_OH'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=0,
            window_size=0)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.FIXED_VECTOR,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=80,
            window_size=0)))
    return ads, run_name

def Experiment_Synthetic_All_Models_W2V(repeats=5):
    bucket = [3,4,5,6,7,8,9]
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Synthetic_All_Models_W2V'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=2)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=16)))
    return ads, run_name

def Experiment_Synthetic_All_Models_T2V(repeats=5):
    bucket = [3,4,5,6,7,8,9]
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Synthetic_All_Models_T2V'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=4)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=2)))
    return ads, run_name

def Experiment_Synthetic_All_Models(repeats=5):
    bucket = [3,4,5,6,7,8,9]
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Synthetic_All_Models'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=0,
            window_size=0)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.FIXED_VECTOR,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=80,
            window_size=0)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=2)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=16)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=4)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=2)))
    
    return ads, run_name     

def Experiment_Batch_Size(repeats=3):
    """
    Setup:
    Run on chosen synthetic datasets 0.45 anomaly percentages \n
    Using most common Onehot and W2V with baseline bucketing settings.\n
    -- \n
    Goal:
    Determine the cost/benifit of batch sizes. Assumption is that lower batchsizes will score better but take longer. Goal is to determine an acceptable middleground. \n
    -- \n 
    Conclusion:
    Batch sizes seem to effect different encoding types differently. With One-hot favoring small batchsizes and W2V larger ones. 
    This is mainly due to One-Hot being able to learn the order perspective effectively on a low batchsize while W2v fails to.

    Workload increases overall when increasing the batch size
    Order decreases overall when increasing the batch size
    All other perspectives stay stable.
    """
    bucket = [3,4,5,6,7,8,9] # As high degree of bucketing will potentially cause issues when not using prefixes
    prefix = True

    ads_2, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=2, bucket=bucket, prefix=prefix)
    ads_4, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=4, bucket=bucket, prefix=prefix)
    ads_8, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=8, bucket=bucket, prefix=prefix)
    ads_16, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=16, bucket=bucket, prefix=prefix)

    run_name = 'Experiment_Batch_Size'
    return ads_2 + ads_4 + ads_8 + ads_16, run_name

def Experiment_Prefix(repeats=3):
    """
    Setup:
    Run on chosen synthetic datasets 0.45 anomaly percentages. \n
    Using most common Onehot and W2V with baseline batch settings and no bucketing.\n
    -- \n
    Goal:
    Determine if the use of prefixes is beneficial. \n
    -- \n 
    Conclusion:
    TBD
    """
    batch_size = 8
    bucket = None # As high degree of bucketing will potentially cause issues when not using prefixes

    ads_no_prefix, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=batch_size, bucket=bucket, prefix=False)
    ads_prefix, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=batch_size, bucket=bucket, prefix=True)
    bucket = [3,4,5,6,7,8,9] # Testing the effect of bucketing on prefixes
    ads_prefix_bucket, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=batch_size, bucket=bucket, prefix=True)
    ads_prefix_bucket, _ = Experiment_Anomaly_Percentage(repeats=repeats, batch_size=batch_size, bucket=bucket, prefix=False)

    run_name = 'Experiment_Prefix'
    return ads_no_prefix + ads_prefix + ads_prefix_bucket, run_name

def Experiment_Synthetic_Dataset(repeats=3):
    """
    Setup:
    Run on all synthetic datasets 0.45 anomaly percentages \n
    Using most common Onehot and W2V with baseline bucketing and batch settings \n
    -- \n
    Goal:
    Determine synthetic dataset with highest scores (highest seperability of normal and anomalous data) and choose fixed synthetic dataset to then iterate over parameters. \n
    -- \n 
    Conclusion:
    Use Gigantic dataset, as it scores the highest while also have one of the lowest runtimes.
    """
    ads, _ = Experiment_Anomaly_Percentage(repeats=repeats)
    run_name = 'Experiment_Synthetic_Dataset'

    return ads, run_name

def Experiment_Anomaly_Percentage(repeats=1, batch_size=8, bucket=[3,4,5,6,7,8,9], prefix=True):
    """
    Setup:
    Run on medium synthetic dataset 0.1-0.45 anomaly percentages \n
    Using most common Onehot and W2V with baseline bucketing and batch settings \n
    -- \n
    Goal:
    Determine anomaly percentage with highest scores (highest seperability of normal and anomalous data) and choose fixed anomaly percentage to then iterate over datasets. \n
    -- \n 
    Conclusion:
    Use the 0.45 anomaly percentage.
    """
    run_name = 'Experiment_Anomaly_Percentage'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=0,
            window_size=0)))
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=2)))
    
    return ads, run_name 


def Experiment_Finetuning_W2V_Window_Vector_Sizes(repeats=3):
    run_name = 'Experiment_Finetuning_W2V_Window_Vector_Sizes'
    vector_sizes = [10,20,40,80,160]
    window_sizes = [2,4,8,16]
    pre_train_percentage = [0]

    ads = []
    combinations = list(itertools.product(vector_sizes, window_sizes, pre_train_percentage))
    for _ in range(repeats):
        for combination in combinations:
            vector_size, window_size, pre_train_percentage = combination

            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=8, 
                prefix=True, 
                bucket_boundaries = [3,4,5,6,7,8,9],
                categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=pre_train_percentage,
                vector_size = vector_size,
                window_size = window_size)))
            
            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=8, 
                prefix=True, 
                bucket_boundaries = [3,4,5,6,7,8,9],
                categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=pre_train_percentage,
                vector_size = vector_size,
                window_size = window_size)))
    
    return ads,run_name

def Experiment_Finetuning_T2V_Window_Vector_Sizes(repeats=3):
    run_name = 'Experiment_Finetuning_T2V_Window_Vector_Sizes'
    vector_sizes = [10,20,40,80,160]
    window_sizes = [2,4,8,16]
    pre_train_percentage = [0]

    batch_size = 8
    bucket_boundaries = [3,4,5,6,7,8,9]
    prefix=True

    ads = []
    combinations = list(itertools.product(vector_sizes, window_sizes, pre_train_percentage))
    for _ in range(repeats):
        for combination in combinations:
            vector_size, window_size, pre_train_percentage = combination

            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=batch_size, 
                prefix=prefix, 
                bucket_boundaries = bucket_boundaries,
                categorical_encoding=EncodingCategorical.TRACE_2_VEC_ATC,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=pre_train_percentage,
                vector_size = vector_size,
                window_size = window_size)))
            
            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=batch_size, 
                prefix=prefix, 
                bucket_boundaries = bucket_boundaries,
                categorical_encoding=EncodingCategorical.TRACE_2_VEC_C,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=pre_train_percentage,
                vector_size = vector_size,
                window_size = window_size)))
    
    return ads,run_name

def Experiment_Finetuning_Fixed_Vector_Vector_Sizes(repeats=3):
    run_name = 'Experiment_Finetuning_Fixed_Vector_Vector_Sizes'
    vector_sizes =  [10,20,40,80,160,240,320]
    batch_sizes = [8]
    buckets = [[3,4,5,6,7,8,9]]

    prefix=True

    ads = []
    combinations = list(itertools.product(vector_sizes, batch_sizes, buckets))
    for _ in range(repeats):
        for combination in combinations:
            vector_size, batch_size, bucket = combination

            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=batch_size, 
                prefix=prefix, 
                bucket_boundaries=bucket,
                categorical_encoding=EncodingCategorical.FIXED_VECTOR,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=0,
                vector_size=vector_size,
                window_size=0)))
    
    return ads,run_name