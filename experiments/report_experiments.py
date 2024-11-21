import numpy as np
from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical

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
    TBD
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

    run_name = 'Experiment_Prefix'
    return ads_no_prefix + ads_prefix, run_name

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
    TBD
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
    TBD
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
            vector_size=200,
            window_size=10)))
    
    return ads, run_name 