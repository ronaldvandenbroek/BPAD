import itertools
from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical

def W2V_gridsearch_vector_window_size():
    run_name = 'W2V_Gridsearch_v3_batch_8_no_pretrain'
    vector_sizes = [250,300] #[20,40,60,80,100,150,200]
    window_sizes = [2,4,6,8,10,12,14]
    pre_train_percentage = [0]

    ads = []
    combinations = list(itertools.product(vector_sizes, window_sizes, pre_train_percentage))
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
        
    # ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=8, 
    #         prefix=True, 
    #         bucket_boundaries = [3,4,5,6,7,8,9],
    #         categorical_encoding=EncodingCategorical.ONE_HOT,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=0,
    #         window_size=0)))
    
    return ads,run_name

def W2V_gridsearch_bucket_batchsize():
    run_name = 'W2V_Gridsearch_v1_bucket_batch_size'
    vector_sizes = [20]
    window_sizes = [2]
    pre_train_percentage = [0]
    batch_sizes = [2,4,8,16]
    buckets = [None, [5,8], [3,4,5,6,7,8,9]]

    ads = []
    combinations = list(itertools.product(vector_sizes, window_sizes, pre_train_percentage, batch_sizes, buckets))
    for combination in combinations:
        vector_size, window_size, pre_train_percentage, batch_size, bucket = combination

        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=pre_train_percentage,
            vector_size=vector_size,
            window_size=window_size)))
        
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=batch_size, 
                prefix=True, 
                bucket_boundaries=bucket,
                categorical_encoding=EncodingCategorical.ONE_HOT,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=0,
                vector_size=0,
                window_size=0)))
    
    return ads,run_name

def W2V_finetuned():
    run_name = 'W2V_finetuned'
    vector_sizes = [20]
    window_sizes = [2]
    pre_train_percentage = [0]
    batch_sizes = [8] #RCVDB: TODO finetune batch size and buckets
    buckets = [[3,4,5,6,7,8,9]]

    ads = []
    combinations = list(itertools.product(vector_sizes, window_sizes, pre_train_percentage, batch_sizes, buckets))
    for combination in combinations:
        vector_size, window_size, pre_train_percentage, batch_size, bucket = combination

        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=pre_train_percentage,
            vector_size=vector_size,
            window_size=window_size)))
        
    # ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=8, 
    #         prefix=True, 
    #         bucket_boundaries = [3,4,5,6,7,8,9],
    #         categorical_encoding=EncodingCategorical.ONE_HOT,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=0,
    #         window_size=0)))
    
    return ads,run_name

def W2V_pretrain():
    run_name = 'W2V_pretrain'
    vector_sizes = [20]
    window_sizes = [2]
    pre_train_percentages = [0,0.01,0.05,0.1,0.5]
    batch_sizes = [8] #RCVDB: TODO finetune batch size and buckets
    buckets = [[3,4,5,6,7,8,9]]
    repeats = 5
    
    ads = []
    for pre_train_percentage in pre_train_percentages:
        for _ in range(repeats):
            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=batch_sizes[0], 
                prefix=True, 
                bucket_boundaries=buckets[0],
                categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=pre_train_percentage,
                vector_size=vector_sizes[0],
                window_size=window_sizes[0])))
    
    return ads,run_name

def W2V_no_averaging():
    run_name = 'W2V_no_averaging'
    vector_sizes = [20]
    window_sizes = [2]
    pre_train_percentage = [0]
    batch_sizes = [8] #RCVDB: TODO finetune batch size and buckets
    buckets = [[3,4,5,6,7,8,9]]

    ads = []
    combinations = list(itertools.product(vector_sizes, window_sizes, pre_train_percentage, batch_sizes, buckets))
    for combination in combinations:
        vector_size, window_size, pre_train_percentage, batch_size, bucket = combination

        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.WORD_2_VEC_C,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=pre_train_percentage,
            vector_size=vector_size,
            window_size=window_size)))
        
    # ads.append(dict(ad=DAE, fit_kwargs=dict( 
    #         batch_size=8, 
    #         prefix=True, 
    #         bucket_boundaries = [3,4,5,6,7,8,9],
    #         categorical_encoding=EncodingCategorical.ONE_HOT,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=0,
    #         window_size=0)))
    
    return ads,run_name

