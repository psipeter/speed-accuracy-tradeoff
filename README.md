# A Spiking Neural Model of Decision Making and the Speed-Accuracy Tradeoff

The speed-accuracy tradeoff (SAT) is a well-characterized feature of decision making (DM) tasks in which fast decisions come at the expense of accurate performance. Despite the successes of computational models like the drift diffusion model, there is a scarcity of biologically-plausible, mechanistic accounts of how the brain realizes the cognitive operations required for DM, limiting our ability to unify neurological and computational accounts of the SAT. We present a spiking neural network model that functionally extends the evidence accumulation framework using an architecture based on the functional neuroanatomy of the human brain. We apply the model to both perceptual and cognitive DM tasks, investigate multiple experimental contexts within each task, and validate model performance by comparing it to neural and behavioral data from monkeys and humans. We find that our model (a) reproduces the response time (RT) distributions of individuals; (b) generalizes across experimental contexts, including the number of choice alternatives, speed or accuracy instructions, and novel difficulty conditions; and (c) predicts accuracy data, despite being trained only on RT data. Putting these results together, we show that our model (1) recreates observed patterns of spiking neural activity, (2) explains how individual differences in RT and accuracy arise from parametric differences in a biologically-plausible neural network, and (3) captures characteristic SAT curves in both perceptual and cognitive DM tasks. We conclude by discussing the neural and cognitive plausibility of our model, comparing our work to other neural and non-neural DM models, and suggesting directions for future work.

## Installation
 > pip install numpy scipy matplotlib seaborn pandas jupyter nengo optuna mysql-connector
 
## Code
 - **model.py** contains the model definition for the spiking neural network and for the sampling modules
 - **dynamics.ipynb** contains code for running the core model and inspecting its dynamics; generates Figures 2-7
 - **[name]_compare.ipynb** contains code for simulating many parameterized instances of the model, organizing the corresponding empirical data into pandas dataframes, and comparing simulated and empirical data
 - **/optimize** contains helper files for running the optimization software on the Compute Canada cluster

## Folders
 - **/data** contains all raw empirical datafiles; all reformatted empirical data, and all simulated data, are saved here
 - **/plots** contains all plots output by the jupyter notebooks
