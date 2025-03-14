from baseline.comb.comb import COMB
from novel.transformer.transformer import Transformer
from utils.enums import EncodingCategorical, EncodingNumerical, Perspective

def Experiment_Offline_COMB(dataset_folder='batching-base'):
    n_epochs = [1, 20]

    ads = []
    for epoch in n_epochs:
        ads.append(dict(ad=COMB, fit_kwargs=dict( 
            batch_size=64,
            n_epochs=epoch, 
            prefix=False, 
            categorical_encoding=EncodingCategorical.TOKENIZER,
            numerical_encoding=EncodingNumerical.MIN_MAX_SCALING
            )))
    run_name = 'Offline_Training'

    return ads, run_name, dataset_folder

def Experiment_Offline_MP_Former(repeats=1, dataset_folder='batching-base'):
    # Static configuration
    prefix = True
    
    run_name = 'Offline_Training'

    ads = []

    # Ablation study
    event_positional_encoding = False
    multi_task = True
    use_prefix_errors = True

    # Debugging/Development
    case_limit = None # Capping the training data for development purposes, set to None for full training
    debug_logging = False
    
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