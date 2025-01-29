import itertools
from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical

def ELMo_finetuned(
        run_name='ELMo_Synthetic',
        vector_size=1024,
        batch_size=8,
        bucket=[3,4,5,6,7,8,9],
        repeats=2,
        prefix=True):

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=bucket,
            categorical_encoding=EncodingCategorical.ELMO,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=vector_size,
            window_size=0)))
        
    return ads,run_name


