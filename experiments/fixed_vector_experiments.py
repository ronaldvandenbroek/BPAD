import itertools
from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical

def Fixed_Vector_gridsearch_vector_sizes():
    run_name = 'Fixed_vector_gridsearch_vector_sizes'
    vector_sizes =  [10,20,30,40,50,60,70,80,90,100]
    batch_sizes = [8]
    buckets = [[3,4,5,6,7,8,9]]

    ads = []
    combinations = list(itertools.product(vector_sizes, batch_sizes, buckets))
    for combination in combinations:
        vector_size, batch_size, bucket = combination

        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.FIXED_VECTOR,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=vector_size,
            window_size=0)))
    
    return ads,run_name

def Fixed_Vector_finetuned():
    run_name = 'Fixed_vector_finetuned'
    vector_sizes = [20]
    pre_train_percentage = [0]
    batch_sizes = [8] #RCVDB: TODO finetune batch size and buckets
    buckets = [[3,4,5,6,7,8,9]]

    ads = []
    combinations = list(itertools.product(vector_sizes, batch_sizes, buckets))
    for combination in combinations:
        vector_size, batch_size, bucket = combination

        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=True, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.FIXED_VECTOR,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=vector_size,
            window_size=0)))
    
    return ads,run_name