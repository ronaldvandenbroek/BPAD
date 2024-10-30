import itertools
from novel.dae.dae import DAE
from utils.enums import EncodingCategorical, EncodingNumerical

def W2V_gridsearch_vector_window_size():
    run_name = 'W2V_Gridsearch_v3_batch_8_no_pretrain'
    w2v_vector_sizes = [20,40,60,80,100,150,200]
    w2v_window_size = [2,4,6,8,10]
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
    
    return ads_w2v_embedding_search,run_name

def W2V_finetuned():
    run_name = 'W2V_Gridsearch_v3_batch_8_no_pretrain'
    w2v_vector_sizes = [20]
    w2v_window_size = [2]
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
    
    return ads_w2v_embedding_search,run_name

