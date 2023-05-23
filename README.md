# A Spiking Neural Model of Decision Making and the Speed-Accuracy Tradeoff

## Installation
 > pip install numpy scipy matplotlib seaborn pandas jupyter nengo nni "typeguard<3" filelock==3.10
 
## Code
 - **model.py** contains code for building and simulating the neural network
 - **fit_params.py** contains code run by NNI to run all trials for a given set of parameters
 - **start_fitting.py** contains wrapper code for defining NNI search space and running the optimization
 - **collect.py** should be run after the NNI experiments complete, to collect the data and identify the best parameter values
 - **dynamics.ipynb** plots Figures 2-3
 - **individual_behavior.py** plots Figure 4

## Running
 - edit start_fitting.py to specify the hyperparameter search
 - run **python start_fitting.py** to generate the config and parameter files, and automatically start an NNI experiment in a subprocess
 - OR comment out the last two lines in start_fitting.py, run it to generate the config and parameter files, then manually run **nnictl create --config config.yaml --port 8080** to start the NNI experiment



