#!/bin/bash
#SBATCH --job-name=test_python
#SBATCH --output=test_python-%j.log
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=00:05:00

# Load modules
module purge
module load Python/3.9.5-GCCcore-10.3.0
module load Miniconda3/24.1.2-0

# Activate conda
eval "$(conda shell.bash activate)"
conda activate rcvdb-thesis-bpad

cd $HOME/Thesis/BPAD/

python main_unsup.py
