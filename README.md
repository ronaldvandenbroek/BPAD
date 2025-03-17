# OMP-AD framework

![Framework Architecture](/readme_figures/framework-architecture.png)

This repositiory is part of the **O**nline **M**ulti-**P**erspective **A**nomaly **D**etection framework proposed in the master thesis 'Exploring Multi-Perspective Anomaly Detection with Transformers in Process-Streams' by [Ronald van den Broek](https://github.com/ronaldvandenbroek), supervised by dr. ing. Marwan Hassani. 

The full OMP-AD framework consists of two repositories:
* [OMP-AD-Datagen](https://github.com/ronaldvandenbroek/O-MP-AD-datagen) is responsible for generating the datasets corresponding to the extended setting proposed in the thesis, implementing the *arrival-time* and *workload* anomalies. 
* [OMP-AD-Models](https://github.com/ronaldvandenbroek/O-MP-AD-models) contains the implementation and evaluation of the proposed novel *BP-DAE* and *MP-Former* models. 

## OMP-AD-Models
This repository is an extention of the [BPAD](https://github.com/guanwei49/BPAD) framework, extended to execute the extended setting proposed within the thesis. It contains the novel *BP-DAE* and *MP-Former* anomaly detection models.

### Folder overview
* **analysis/**
    * To analyse the run experiments with the [result_analysis.ipynb](analysis/result_analysis.ipynb) notebook, which results are saved in **analysis/results**.

    <!-- * **analysis/prelimenaries** contain additional analysis notebooks  -->

* **baseline/** 
    * Contains the state-of-the-art offline [COMB](https://github.com/guanwei49/COMB) model modified to work in the extended setting.

* **eventlogs_pregenerated/**
    * Contains all generated datasets from the [OMP-AD-Datagen](https://github.com/ronaldvandenbroek/O-MP-AD-datagen) repository. To utilize these datasets they can be copied to the **eventlogs/** folder.

* **experiments/**
    * Contains all the experiment configurations used within the thesis.

* **novel/** 
    * **novel/dae/** Contains the novel BP-DAE model based on the [OAE](https://github.com/zyrako4/sequence-online-ad) model.

    * **novel/transformer/** Contains the novel MP-Former model inspired by the [MTLFormer](https://github.com/jiaojiaowang1992/MTLFormer) model.

* **scripts_hpc/**
    * Contains the script files utilized for running the experiments on a computation cluster.
   
## Setup
### Environment
To install the required packages the `conda` environment `rcvdb-thesis-bpad` should be installed. `conda` can be utilized by installing [Miniconda](https://conda.io/miniconda.html).

The environments can be installed by running the following commands:

```
conda env remove --name "rcvdb-thesis-bpad"
conda activate rcvdb-thesis-bpad
```

### Run experimental setup
The experimental setup can be run in two seperate ways:

To run the experiments locally, the [main.py](main.py) can be configured and run as a normal python program. 
```python
    run_local=True # Set to true of running local, false otherwise
    if run_local:
        experiment = 'Experiment_Example'
        dataset_folder = 'dataset_example'
        repeats = 1 # Number of times the experiment is repeated
```

Alternatively the experiments can be run on a computation cluster, depending on the cluster the scipt files might differ, however examples can be found in the **scripts_hpc/** folder. Most importantly `run_local=False` should be set in `main.py`.

After running the experiments the results should be moved to **results/raw/**, which is done automatically if running locally. Then the [analysis/result_analysis.ipynb](analysis/result_analysis.ipynb) notebook can be run to analyze the results: 

```python
directories = [ # Results foldername in the results directory
    'Experiment_Example' 
]

recalculate = False # Enable/Disable caching of results
```

## License
This repo is published under the GNU GENERAL PUBLIC LICENSE following BiNet [1]. 

## Sources
1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning, 2018](https://doi.org/10.1007/978-3-319-98648-7_16)