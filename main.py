import os
import warnings
import argparse

from experiments.report_experiments import Experiment_Anomaly_Percentage, Experiment_Batch_Size, Experiment_Finetuning_Fixed_Vector_Vector_Sizes, Experiment_Finetuning_T2V_Window_Vector_Sizes, Experiment_Finetuning_W2V_Window_Vector_Sizes, Experiment_Prefix, Experiment_Real_World_T2V_ATC, Experiment_Real_World_T2V_C, Experiment_Real_World_W2V_ATC, Experiment_Synthetic_All_Models, Experiment_Synthetic_All_Models_FV_OH, Experiment_Synthetic_All_Models_T2V, Experiment_Synthetic_All_Models_W2V, Experiment_Synthetic_Dataset
from main_anomaly_detection import execute_runs, prepare_datasets

# RCVDB: Supressing Sklearn LabelEncoder InconsistentVersionWarning as this seems an internal package issue
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Anomaly Detection on Process Streams")
    parser.add_argument('--experiment', type=str, required=False)
    parser.add_argument('--repeats', type=int, required=False)
    parser.add_argument('--experiment_name', type=str, required=False)
    args = parser.parse_args()

    seed=2024
    dataset_names = prepare_datasets()

    if args.experiment:
        experiment = args.experiment
    else:
        experiment = "None"

    if args.repeats:
        repeats = args.repeats
    else:
        repeats = 1

    # If running main locally without passing arguments set run_local to true and configure the run manually 
    run_local=False
    if run_local:
        experiment = "Experiment_Real_World_T2V_C"
        repeats = 3
        # run_name = "Custom name"

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

    # If a custom name is specified
    if args.experiment_name:
        run_name = args.experiment_name

    if ads is not None and run_name is not None:
        execute_runs(dataset_names, ads, run_name, seed)