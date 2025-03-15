#!/bin/bash

#SBATCH --job-name=synthetic_COMB
#SBATCH --output=scripts_hpc/log/test_python-%j.log
#SBATCH --partition=tue.gpu.q         
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --gpus=1                     

# Load modules
module purge
module load Python/3.9.5-GCCcore-10.3.0
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Miniconda3/24.1.2-0
module load GCC/12.2.0

source /sw/rl8/zen/app/Miniconda3/24.1.2-0/etc/profile.d/conda.sh

conda activate rcvdb-thesis-bpad

cd $HOME/Thesis/BPAD/

python main.py --experiment "Experiment_Offline_COMB" --repeats 1 --experiment_name "Experiment_Offline_COMB" --dataset_folder "all_datasets_synthetics_part_1"
# python main.py --experiment "Experiment_Offline_COMB" --repeats 1 --experiment_name "Experiment_Offline_COMB" --dataset_folder "all_datasets_synthetics_part_2"
