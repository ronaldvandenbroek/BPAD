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

def Experiment_Synthetic_Transformer(repeats=1, dataset_folder = 'all_datasets_synthetics'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }

    # Ablation study
    event_positional_encoding = True
    multi_task = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Synthetic'
    
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer', 
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

def Experiment_Transformer_Prefix_Store(repeats=1, dataset_folder = 'transformer_debug_synthetic'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }

    # Ablation study
    event_positional_encoding = True
    multi_task = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Prefix_Store'
    
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

def Experiment_Transformer_Event_Positional_Encoding(repeats=1, dataset_folder = 'transformer_debug_synthetic'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }

    # Ablation study
    multi_task = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Event_Positional_Encoding'
    
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_event_pos_enc_true', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=True,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_event_pos_enc_false',  
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=False,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit,
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder

def Experiment_Transformer_Event_Multi_Task(repeats=1, dataset_folder = 'transformer_debug_synthetic'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    perspective_weights = {
        Perspective.ORDER: 1.0,
        Perspective.ATTRIBUTE: 1.0,
        Perspective.ARRIVAL_TIME: 1.0,
        Perspective.WORKLOAD: 1.0,
    }

    # Ablation study
    event_positional_encoding = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Event_Multi_Task'
    
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_multi_task_true', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=True,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_multi_task_false',  
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights=perspective_weights,
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=False,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit,
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder


def Experiment_Transformer_Perspective_Weights_Order(repeats=1, dataset_folder = 'transformer_debug_synthetic'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    
    # Ablation study
    event_positional_encoding = True
    use_prefix_errors = True
    multi_task=True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Perspective_Weights_v2'
    
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_order_only', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 0,
                Perspective.ARRIVAL_TIME: 0,
                Perspective.WORKLOAD: 0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_order_2', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 2.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_order_4', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 4.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_order_8', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 8.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder

def Experiment_Transformer_Perspective_Weights_Attribute(repeats=1, dataset_folder = 'transformer_debug_synthetic'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    
    # Ablation study
    event_positional_encoding = True
    use_prefix_errors = True
    multi_task=True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Perspective_Weights_v2'
    
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_attribute_only', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 0,
                Perspective.WORKLOAD: 0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_attribute_2', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 2.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_attribute_4', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 4.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_attribute_8', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 8.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder

def Experiment_Transformer_Perspective_Weights_Arrival_Time(repeats=1, dataset_folder = 'transformer_debug_synthetic'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    
    # Ablation study
    event_positional_encoding = True
    use_prefix_errors = True
    multi_task=True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Perspective_Weights_v2'
    
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_arrival_time_only', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 0,
                Perspective.ATTRIBUTE: 0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_arrival_time_2', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 2.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_arrival_time_4', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 4.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_attribute_8', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 8.0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder

def Experiment_Transformer_Perspective_Weights_Workload(repeats=1, dataset_folder = 'transformer_debug_synthetic'):
    # Static configuration
    batch_size = 32 #8
    prefix = True
    categorical_encoding = EncodingCategorical.TOKENIZER
    numerical_encoding = EncodingNumerical.MIN_MAX_SCALING
    
    # Ablation study
    event_positional_encoding = True
    use_prefix_errors = True
    multi_task = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False

    run_name = 'Experiment_Transformer_Perspective_Weights_v2'
    
    ads = []
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_workload_only',
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 0,
                Perspective.ATTRIBUTE: 0,
                Perspective.ARRIVAL_TIME: 0,
                Perspective.WORKLOAD: 1.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_workload_2', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 2.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_workload_4', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 4.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    for _ in range(repeats):
        ads.append(dict(ad=Transformer, fit_kwargs=dict(
            model_name='transformer_workload_8', 
            prefix=prefix,
            batch_size=batch_size,
            categorical_encoding=categorical_encoding, # Internally the model uses W2V encoding vector_size is determined by dim_model
            numerical_encoding=numerical_encoding,
            perspective_weights = {
                Perspective.ORDER: 1.0,
                Perspective.ATTRIBUTE: 1.0,
                Perspective.ARRIVAL_TIME: 1.0,
                Perspective.WORKLOAD: 8.0,
            },
            # Ablation study
            event_positional_encoding=event_positional_encoding,
            multi_task=multi_task,
            use_prefix_errors=use_prefix_errors,
            # Debugging/Development
            case_limit=case_limit, 
            debug_logging=debug_logging,
            )))
    return ads, run_name, dataset_folder

