from novel.transformer.transformer import Transformer
from utils.enums import EncodingCategorical, EncodingNumerical, Perspective

def Experiment_Transformer_Debug(repeats=1, dataset_folder='transformer_debug_synthetic'):
    run_name = 'Experiment_Transformer_Config_Debug'
    model_name = f'{dataset_folder}'

    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.5,
        Perspective.ARRIVAL_TIME: 0.2,
        Perspective.WORKLOAD: 0.1,
    }

    # Ablation study
    event_positional_encoding = True
    multi_task = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = 1000 # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name=model_name, 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
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

def Experiment_Transformer_Prefix_Store(repeats=1):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 0.2,
        Perspective.WORKLOAD: 0.1,
    }

    # Ablation study
    event_positional_encoding = True
    multi_task = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Prefix_Store'
    dataset_folder = 'transformer_debug_synthetic'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_prefix_store_true', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=True,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_prefix_store_false',  
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=False,
            # Debugging/Development
            case_limit=case_limit,
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder