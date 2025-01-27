import os
import warnings
import argparse

from experiments.report_experiments import Experiment_Anomaly_Percentage, Experiment_Batch_Size, Experiment_Finetuning_Fixed_Vector_Vector_Sizes, Experiment_Finetuning_T2V_Window_Vector_Sizes, Experiment_Finetuning_W2V_Window_Vector_Sizes, Experiment_Prefix, Experiment_Real_World_T2V_ATC, Experiment_Real_World_T2V_C, Experiment_Real_World_W2V_ATC, Experiment_Synthetic_All_Models, Experiment_Synthetic_All_Models_FV_OH, Experiment_Synthetic_All_Models_T2V, Experiment_Synthetic_All_Models_W2V, Experiment_Synthetic_Dataset
from experiments.transformer_experiments import Experiment_Component_Runtime_Analysis, Experiment_Synthetic_Transformer, Experiment_Transformer_Debug, Experiment_Transformer_Event_Multi_Task, Experiment_Transformer_Event_Positional_Encoding, Experiment_Transformer_Perspective_Weights_Arrival_Time, Experiment_Transformer_Perspective_Weights_Attribute, Experiment_Transformer_Perspective_Weights_Order, Experiment_Transformer_Perspective_Weights_Workload, Experiment_Transformer_Prefix_Store
from main_anomaly_detection import execute_runs, prepare_datasets

# RCVDB: Supressing Sklearn LabelEncoder InconsistentVersionWarning as this seems an internal package issue
from sklearn.exceptions import InconsistentVersionWarning

from utils.fs import FSSave, get_random_id
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Anomaly Detection on Process Streams")
    parser.add_argument('--experiment', type=str, required=False)
    parser.add_argument('--repeats', type=int, required=False)
    parser.add_argument('--experiment_name', type=str, required=False)
    parser.add_argument('--dataset_folder', type=str, required=False)
    args = parser.parse_args()

    seed=2024

    if args.experiment:
        experiment = args.experiment
    else:
        experiment = "None"

    if args.repeats:
        repeats = args.repeats
    else:
        repeats = 1

    if args.dataset_folder:
        dataset_folder = args.dataset_folder
    else:
        dataset_folder = None

    # If running main locally without passing arguments set run_local to true and configure the run manually 
    run_local=False
    if run_local:
        experiment = 'Experiment_Component_Runtime_Analysis' #'Experiment_Transformer_Debug' #'Experiment_Transformer_Prefix_Store' #, 'Experiment_Transformer_Debug' #'Experiment_Real_World_T2V_C' #'Experiment_Prefix'
        # dataset_folder = 'transformer_debug_synthetic' #'experiment_real_world_selected_models' #'all_datasets_synthetics'
        dataset_folder = 'transformer_debug_synthetic'
        repeats = 1

    # Configure the experiment based on the provided arguments
    ads = None
    run_name = None
    if experiment == "Experiment_Batch_Size":
        ads, run_name = Experiment_Batch_Size(repeats=repeats)
    elif experiment == "Experiment_Prefix":
        ads, run_name = Experiment_Prefix(repeats=repeats)
    elif experiment == "Experiment_Anomaly_Percentage": # Experiment_Anomaly_Percentage_v2
        ads, run_name = Experiment_Anomaly_Percentage(repeats=repeats)
    elif experiment == "Experiment_Synthetic_Dataset": # Experiment_Synthetic_Dataset_v5
        ads, run_name = Experiment_Synthetic_Dataset(repeats=repeats)

    # Experiments to finetune models
    elif experiment == "Experiment_Finetuning_Fixed_Vector_Vector_Sizes":
        ads, run_name = Experiment_Finetuning_Fixed_Vector_Vector_Sizes(repeats=repeats)
    elif experiment == "Experiment_Finetuning_W2V_Window_Vector_Sizes":
        ads, run_name = Experiment_Finetuning_W2V_Window_Vector_Sizes(repeats=repeats)
    elif experiment == "Experiment_Finetuning_T2V_Window_Vector_Sizes":
        ads, run_name = Experiment_Finetuning_T2V_Window_Vector_Sizes(repeats=repeats)

    # Experiments to determine best encoding models on synthetic datasets 
    elif experiment == "Experiment_Synthetic_All_Models_FV_OH":
        ads, run_name = Experiment_Synthetic_All_Models_FV_OH(repeats=repeats)
    elif experiment == "Experiment_Synthetic_All_Models_W2V":
        ads, run_name = Experiment_Synthetic_All_Models_W2V(repeats=repeats)
    elif experiment == "Experiment_Synthetic_All_Models_T2V":
        ads, run_name = Experiment_Synthetic_All_Models_T2V(repeats=repeats)
    elif experiment == "Experiment_Synthetic_All_Models":
        ads, run_name = Experiment_Synthetic_All_Models(repeats=repeats)


    # Experiments to determine best encoding models on synthetic datasets 
    elif experiment == "Experiment_Real_World_T2V_C":
        ads, run_name = Experiment_Real_World_T2V_C(repeats=repeats)
    elif experiment == "Experiment_Real_World_T2V_ATC":
        ads, run_name = Experiment_Real_World_T2V_ATC(repeats=repeats)
    elif experiment == "Experiment_Real_World_W2V_ATC":
        ads, run_name = Experiment_Real_World_W2V_ATC(repeats=repeats)

    # Experiments to test transformer components
    elif experiment == "Experiment_Transformer_Debug":
        ads, run_name, dataset_folder = Experiment_Transformer_Debug(repeats=repeats, dataset_folder=dataset_folder)
    # Ablation Studies
    elif experiment == "Experiment_Transformer_Prefix_Store":
        ads, run_name, dataset_folder = Experiment_Transformer_Prefix_Store(repeats=repeats, dataset_folder=dataset_folder)
    elif experiment == "Experiment_Transformer_Event_Positional_Encoding":
        ads, run_name, dataset_folder = Experiment_Transformer_Event_Positional_Encoding(repeats=repeats, dataset_folder=dataset_folder)
    elif experiment == "Experiment_Transformer_Event_Multi_Task":
        ads, run_name, dataset_folder = Experiment_Transformer_Event_Multi_Task(repeats=repeats, dataset_folder=dataset_folder)
    elif experiment == "Experiment_Transformer_Perspective_Weights_Order":
        ads, run_name, dataset_folder = Experiment_Transformer_Perspective_Weights_Order(repeats=repeats, dataset_folder=dataset_folder)
    elif experiment == "Experiment_Transformer_Perspective_Weights_Attribute":
        ads, run_name, dataset_folder = Experiment_Transformer_Perspective_Weights_Attribute(repeats=repeats, dataset_folder=dataset_folder)
    elif experiment == "Experiment_Transformer_Perspective_Weights_Arrival_Time":
        ads, run_name, dataset_folder = Experiment_Transformer_Perspective_Weights_Arrival_Time(repeats=repeats, dataset_folder=dataset_folder)
    elif experiment == "Experiment_Transformer_Perspective_Weights_Workload":
        ads, run_name, dataset_folder = Experiment_Transformer_Perspective_Weights_Workload(repeats=repeats, dataset_folder=dataset_folder)

    # Full Dataset Runs
    elif experiment == "Experiment_Synthetic_Transformer":
        ads, run_name, dataset_folder = Experiment_Synthetic_Transformer(repeats=repeats, dataset_folder=dataset_folder)

    elif experiment == "Experiment_Component_Runtime_Analysis":
        ads, run_name, dataset_folder = Experiment_Component_Runtime_Analysis(repeats=repeats, dataset_folder=dataset_folder)    

    # If a custom name is specified
    if args.experiment_name:
        run_name = args.experiment_name

    print(f"Running experiment: {experiment}")
    print(f'Dataset folder: {dataset_folder}')
    dataset_names = prepare_datasets(dataset_folder)

    run_name = f"{run_name}_{get_random_id()}"

    if ads is not None and run_name is not None:
        execute_runs(dataset_names, ads, run_name, dataset_folder, seed)