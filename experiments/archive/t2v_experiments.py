import itertools
from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical

def T2V_finetuned(
        run_name='Trace2Vec_Synthetic',
        vector_sizes=[20,200],
        window_sizes=[2,10],
        batch_sizes=[8],
        buckets=[[3,4,5,6,7,8,9]],
        repeats=2,
        prefix=True):

    ads = []
    combinations = list(itertools.product(vector_sizes, window_sizes, batch_sizes, buckets))

    for combination in combinations:
        vector_size, window_size, batch_size, bucket = combination        
        for _ in range(repeats):
            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=batch_size, 
                prefix=prefix, 
                bucket_boundaries=bucket,
                categorical_encoding=EncodingCategorical.TRACE_2_VEC_C,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=0,
                vector_size=vector_size,
                window_size=window_size)))

    for combination in combinations:
        vector_size, window_size, batch_size, bucket = combination

        for _ in range(repeats):
            ads.append(dict(ad=DAE, fit_kwargs=dict( 
                batch_size=batch_size, 
                prefix=prefix, 
                bucket_boundaries=bucket,
                categorical_encoding=EncodingCategorical.TRACE_2_VEC_ATC,
                numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
                pretrain_percentage=0,
                vector_size=vector_size,
                window_size=window_size)))
        
    return ads,run_name


