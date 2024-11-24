#!/bin/bash
#SBATCH --job-name=real_world_cpu
#SBATCH --output=scripts_hpc/log/test_python-%j.log
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --time=04:00:00

# Load modules
module purge
module load Python/3.9.5-GCCcore-10.3.0
module load Miniconda3/24.1.2-0
module load GCC/12.2.0

source /sw/rl8/zen/app/Miniconda3/24.1.2-0/etc/profile.d/conda.sh

conda activate rcvdb-thesis-bpad

cd $HOME/Thesis/BPAD/

python main.py --experiment "Experiment_Real_World_W2V_ATC" --repeats 3 --experiment_name "Experiment_Real_World_W2V_ATC_HPC_JOB"
