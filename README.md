# A Spiking Neurocognitive Model of Decision Making and the Speed-Accuracy Tradeoff

The speed-accuracy tradeoff (SAT) is the tendency for fast decisions to come at the expense of accurate performance. Evidence accumulation models such as drift diffusion (DD) can reproduce a variety of behavioral data related to the SAT, and their parameters have been linked to neural activities in the brain. However, our understanding of how biological neural networks realize the cognitive operations required for evidence accumulation remains incomplete, limiting our ability to unify neurological and computational accounts of the SAT. We address this gap by developing and analyzing a biologically-plausible spiking neural network that implements and extends DD. We apply our model to both perceptual and cognitive tasks, investigate several contextual manipulations, and validate model performance using neural and behavioral data. Behaviorally, we find that our model (a) reproduces individual response time distributions; (b) generalizes across experimental contexts, including the number of choice alternatives, speed or accuracy emphasis, and task difficulty; and (c) predicts accuracy data, despite being trained only on response time data. Neurally, we show that our model (1) recreates observed patterns of spiking neural activity, and (2) exhibits behavioral deficits that are consistent with the deficits observed in elderly individuals with degraded corticostriatal connectivity. More broadly, our model captures characteristic SAT curves across a variety of tasks and contexts, and explains how individual differences in speed and accuracy arise from synaptic weights within a neural network. Our work showcases a method for translating DD parameters directly into connection weights, and demonstrates that simulating such networks permits analyses and predictions that are outside the scope of purely mathematical models. We conclude by discussing the neural and cognitive plausibility of our model, comparing our work to other neural network models, and suggesting directions for future work.

## Installation
 > pip install numpy scipy matplotlib seaborn pandas jupyter nengo optuna mysql-connector
 
## Code
 - **model.py** contains the model definition for the spiking neural network and for the sampling modules
 - **dynamics.ipynb** contains code for running the core model and inspecting its dynamics; generates Figures 2-7
 - **[name]_compare.ipynb** contains code for simulating many parameterized instances of the model, organizing the corresponding empirical data into pandas dataframes, and comparing simulated and empirical data
 - **/optimize** contains helper files for running the optimization software on the Compute Canada cluster

## Folders
 - **/data** contains all reformatted empirical data and saved simulated data. Raw empirical data should be requested by contacting the corresponding authors of the original papers.
 - **/plots** contains all plots output by the jupyter notebooks
