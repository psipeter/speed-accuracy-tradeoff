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





