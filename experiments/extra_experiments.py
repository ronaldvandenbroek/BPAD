from novel.dae.dae import DAE
from novel.transformer.transformer import Transformer
from utils.enums import EncodingCategorical, EncodingNumerical, Perspective

def Experiment_Component_Runtime_Analysis(repeats=1, dataset_folder='transformer_debug_synthetic'):
    # Static configuration
    prefix = True
    
    # Ablation study
    event_positional_encoding = True
    multi_task = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Component_Runtime_Analysis'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict(
            model_name='DAE', 
            batch_size=8, 
            prefix=prefix, 
            bucket_boundaries=[3,4,5,6,7,8,9],
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=2)))
    
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer', 
            prefix=prefix,
            batch_size=32,
            categorical_encoding=EncodingCategorical.TOKENIZER, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder

def Experiment_Online_Viablity_DAE(repeats=1, dataset_folder='transformer_debug_synthetic'):
    # Static configuration
    prefix = True
    
    run_name = 'Online_Viablity_DAE'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict(
            model_name='BP-DAE', 
            batch_size=8,
            online_training=True, 
            prefix=prefix, 
            bucket_boundaries=[3,4,5,6,7,8,9],
            categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            pretrain_percentage=0,
            vector_size=160,
            window_size=2)))
    return ads, run_name, dataset_folder


def Experiment_Online_Viablity_Transformer(repeats=1, dataset_folder='transformer_debug_synthetic'):
    # Static configuration
    prefix = True
    
    run_name = 'Online_Viablity_Transformer'

    ads = []

    # Ablation study
    event_positional_encoding = True
    multi_task = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False
    
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='MP-Former',
            online_training=True, 
            prefix=prefix,
            batch_size=32,
            categorical_encoding=EncodingCategorical.TOKENIZER, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder



def Experiment_Offline_Training(repeats=1, dataset_folder='transformer_debug_synthetic'):
    # Static configuration
    prefix = True
    
    run_name = 'Offline_Training'

    ads = []
    # for _ in range(repeats):
    #     ads.append(dict(ad=DAE, fit_kwargs=dict(
    #         model_name='DAE-offline', 
    #         batch_size=8,
    #         online_training=False, 
    #         prefix=prefix, 
    #         bucket_boundaries=[3,4,5,6,7,8,9],
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=160,
    #         window_size=2)))
    #     ads.append(dict(ad=DAE, fit_kwargs=dict(
    #         model_name='DAE-online', 
    #         batch_size=8,
    #         online_training=True, 
    #         prefix=prefix, 
    #         bucket_boundaries=[3,4,5,6,7,8,9],
    #         categorical_encoding=EncodingCategorical.WORD_2_VEC_ATC,
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         pretrain_percentage=0,
    #         vector_size=160,
    #         window_size=2)))

    # Ablation study
    event_positional_encoding = True
    multi_task = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False
    
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }
    # for _ in range(repeats):
    #     ads.append(dict(ad=Transformer, fit_kwargs=dict(
    #         model_name='transformer-offline',
    #         online_training=False, 
    #         prefix=prefix,
    #         batch_size=32,
    #         categorical_encoding=EncodingCategorical.TOKENIZER, # Internally the model uses W2V encoding vector_size is determined by dim_model
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         perspective_weights=perspective_weights,
    #         # Ablation study
    #         event_positional_encoding=event_positional_encoding,
    #         multi_task=multi_task,
    #         use_prefix_errors=use_prefix_errors,
    #         # Debugging/Development
    #         case_limit=case_limit, 
    #         debug_logging=debug_logging,
    #         )))
    #     ads.append(dict(ad=Transformer, fit_kwargs=dict(
    #         model_name='transformer-online',
    #         online_training=True, 
    #         prefix=prefix,
    #         batch_size=32,
    #         categorical_encoding=EncodingCategorical.TOKENIZER, # Internally the model uses W2V encoding vector_size is determined by dim_model
    #         numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
    #         perspective_weights=perspective_weights,
    #         # Ablation study
    #         event_positional_encoding=event_positional_encoding,
    #         multi_task=multi_task,
    #         use_prefix_errors=use_prefix_errors,
    #         # Debugging/Development
    #         case_limit=case_limit, 
    #         debug_logging=debug_logging,
    #         )))
        
    perspective_weights_oa = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 0,
        Perspective.WORKLOAD: 0,
    }
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer-oa-offline',
            online_training=False, 
            prefix=prefix,
            batch_size=32,
            categorical_encoding=EncodingCategorical.TOKENIZER, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            perspective_weights=perspective_weights_oa,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer-oa-online',
            online_training=True, 
            prefix=prefix,
            batch_size=32,
            categorical_encoding=EncodingCategorical.TOKENIZER, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            perspective_weights=perspective_weights_oa,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder