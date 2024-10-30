from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical


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