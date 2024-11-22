https://hpcwiki.tue.nl/documentation/steps/access/

https://hpcwiki.tue.nl/documentation/software/recipes/python/

<!-- Todo create as script -->
``` module purge
module load Python/3.9.5-GCCcore-10.3.0
module load Miniconda3/24.1.2-0
eval "$(conda shell.bash activate)"
conda env create -f generation/environment.yml
conda activate rcvdb-thesis-bpad
 ```