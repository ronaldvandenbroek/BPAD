from novel.transformer.transformer import Transformer
from utils.enums import EncodingCategorical, EncodingNumerical

def Experiment_Transformer_Debug(repeats=1):
    batch_size = 8
    prefix = True
    run_name = 'Experiment_Transformer_Debug'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict( 
            batch_size=batch_size, 
            prefix=prefix, 
            bucket_boundaries=None,
            categorical_encoding=EncodingCategorical.ONE_HOT,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=0,
            window_size=0)))
    return ads, run_name