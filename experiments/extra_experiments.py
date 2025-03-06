from novel.dae.dae import DAE
from novel.transformer.transformer import Transformer
from utils.enums import EncodingCategorical, EncodingNumerical, Perspective

def Template_Experiment_DAE(repeats=1, batch_size=8, bucket=[3,4,5,6,7,8,9], prefix=True):
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
            vector_size=160,
            window_size=2)))
    return ads

def Template_Experiment_Transformer(repeats=1, batch_size=32, prefix=True, perspective_weights=None, event_positional_encoding=True, multi_task=True, use_prefix_errors=True, case_limit=None, debug_logging=False):
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            prefix=prefix,
            batch_size=batch_size,
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
    return ads

def Experiment_DAE_Batch_Size(repeats=3, dataset_folder='transformer_debug_synthetic'):
    bucket = [3,4,5,6,7,8,9] # As high degree of bucketing will potentially cause issues when not using prefixes
    prefix = True
    batch_sizes = [1,2,4,8,16]
    ads = []
    for batch_size in batch_sizes:
        ads += Template_Experiment_DAE(repeats=repeats, batch_size=batch_size, bucket=bucket, prefix=prefix)

    run_name = 'Batch_Size_BP-DAE'
    return ads, run_name, dataset_folder

def Experiment_Transformer_Batch_Size(repeats=3, dataset_folder='transformer_debug_synthetic'):
    prefix = True
    batch_sizes = [8,16,32,64,128]
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }
    ads = []
    for batch_size in batch_sizes:
        ads += Template_Experiment_Transformer(repeats=repeats, batch_size=batch_size, prefix=prefix, perspective_weights=perspective_weights)

    run_name = 'Batch_Size_MP-Former'
    return ads, run_name, dataset_folder

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

    if 'real' in dataset_folder:
        bucket_boundaries = [10,20,30,40,60,80]
    else:
        bucket_boundaries=[3,4,5,6,7,8,9]

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=DAE, fit_kwargs=dict(
            model_name='BP-DAE', 
            batch_size=8,
            online_training=True, 
            prefix=prefix, 
            bucket_boundaries=bucket_boundaries,
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