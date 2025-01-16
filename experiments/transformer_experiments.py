from novel.transformer.transformer import Transformer
from utils.enums import EncodingCategorical, EncodingNumerical, Perspective

def Experiment_Transformer_Debug(repeats=1):
    batch_size = 32 #8
    prefix = True
    run_name = 'Experiment_Transformer_Config_Debug'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict( 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=EncodingCategorical.TOKENIZER, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING,
            # Ablation study
            event_positional_encoding=True,
            use_prefix_errors=True,
            # Debugging/Development
            case_limit=1000, # Capping the training data for development purposes, set to None for full training
            debug_logging=False,
            )))
    return ads, run_name

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
    multi_task = False

    # Debugging/Development
    case_limit = 1000 # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Prefix_Store'

    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict( 
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
    return ads, run_name