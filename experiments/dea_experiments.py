from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical

def DAE_finetuned_embedding_batch_size_1():
    run_name = 'DAE_finetuned_embedding_batch_size_1'
    batch_size = 8
    bucket = [3,4,5,6,7,8,9]

    ads = []
    ads.append(dict(ad=DAE, fit_kwargs=dict( 
        batch_size=batch_size, 
        prefix=True, 
        bucket_boundaries=bucket,
        categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
        numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
        pretrain_percentage=0,
        vector_size=200,
        window_size=10)))
    
    return ads, run_name

def DAE_finetuned_embedding():
    run_name = 'DAE_Finetuned_Embedding'
    batch_size = 8
    bucket = [3,4,5,6,7,8,9]
    repeats = 5

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict(
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries = bucket,
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=0,
            window_size=0)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.FIXED_VECTOR,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=50,
            window_size=0)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=2)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=200,
            window_size=10)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=2)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=200,
            window_size=10)))
    
    return ads, run_name


def DAE_gridsearch_batch_bucketing():
    # RCVDB: Configuration to test multi-label anomalies
    # In practice probably more accurrate to have one epoch and a batch size of one to simulate each event arriving seperately
    run_name = 'DAE_Gridsearch'
    ads = [
        dict(ad=DAE, fit_kwargs=dict( 
            batch_size=8, 
            prefix=True, 
            bucket_boundaries = [3,4,5,6,7,8,9],
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0.5,
            vector_size = 100,
            window_size = 10)),
        dict(ad=DAE, fit_kwargs=dict( 
            batch_size=1, 
            prefix=True, 
            bucket_boundaries = None, # [3,4,5,6,7,8,9], #None
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            vector_size = 100,
            window_size = 10)),
        dict(ad=DAE, fit_kwargs=dict(
            batch_size=1, 
            prefix=True, 
            bucket_boundaries = None,
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            vector_size = 100,
            window_size = 10)),
        dict(ad=DAE, fit_kwargs=dict(
            batch_size=1, 
            prefix=True, 
            bucket_boundaries = [5,8], # [3,4,5,6,7,8,9], #None
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            vector_size = 100,
            window_size = 10)),
        dict(ad=DAE, fit_kwargs=dict(
            batch_size=1, 
            prefix=True, 
            bucket_boundaries = [5,8],
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            vector_size = 100,
            window_size = 10)),
        dict(ad=DAE, fit_kwargs=dict( 
            batch_size=8, 
            prefix=True, 
            bucket_boundaries = [3,4,5,6,7,8,9], #None
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            vector_size = 100,
            window_size = 10)),
        dict(ad=DAE, fit_kwargs=dict( 
            batch_size=8, 
            prefix=True, 
            bucket_boundaries = [3,4,5,6,7,8,9],
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            vector_size = 100,
            window_size = 10)),
    ]
    return ads,run_name


def DAE_repeatability_experiment():
    run_name = 'DAE_Repeatability'
    batch_size = 8
    bucket = [3,4,5,6,7,8,9]
    repeats = 10

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=20,
            window_size=2)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=0,
            window_size=0)))
    
    return ads, run_name

def DAE_bpic2015_no_buckets(
        run_name='DAE_bpic2015_no_buckets',
        batch_size=8,
        bucket=None,
        repeats=1,
        prefix=True):
    
    ads = []

    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict(
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries = bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=200,
            window_size=10)))

    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict(
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries = bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=200,
            window_size=10)))

    return ads, run_name  

def DAE_bpic2015(
        run_name='DAE_bpic2015',
        batch_size=8,
        bucket=[20,30,40,50,60],
        repeats=1,
        prefix=False):

    ads = []
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict(
    #         batch_size=batch_size, 
    #         prefix=prefix, 
    #         bucket_boundaries = bucket,
    #         categorical_encoding=EncodingCategorical.ONE_HOT,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=0,
    #         window_size=0)))
        
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=batch_size, 
    #         prefix=prefix, 
    #         bucket_boundaries=bucket,
    #         categorical_encoding=EncodingCategorical.FIXED_VECTOR,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=50,
    #         window_size=0)))
        
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=batch_size, 
    #         prefix=prefix, 
    #         bucket_boundaries=bucket,
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=20,
    #         window_size=2)))
        
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=batch_size, 
    #         prefix=prefix, 
    #         bucket_boundaries=bucket,
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=200,
    #         window_size=10)))
        
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=batch_size, 
    #         prefix=prefix, 
    #         bucket_boundaries=bucket,
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=20,
    #         window_size=2)))
        
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=batch_size, 
    #         prefix=prefix, 
    #         bucket_boundaries=bucket,
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=200,
    #         window_size=10)))
        
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=batch_size, 
    #         prefix=prefix, 
    #         bucket_boundaries=bucket,
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=200,
    #         window_size=10)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=200,
            window_size=10)))
        
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.TRACE_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=200,
            window_size=10)))   
    
    return ads, run_name
