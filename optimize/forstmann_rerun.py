import numpy as np
import nengo
from nengo import Lowpass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.io
import sys
import optuna
import mysql.connector
from scipy.stats import gaussian_kde
from model import DotPerception

def build_network(inputs, nActions=2, nNeurons=50, synapse=0.1, seed=0, ramp=1, threshold=0.3, relative=0,
        max_rates=nengo.dists.Uniform(60, 80), rA=1, spike_filter=0.03, w_accumulator=None, w_threshold=None, w_or_d=None):
    
    net = nengo.Network(seed=seed)
    net.config[nengo.Connection].synapse = 0.03
    net.config[nengo.Probe].synapse = 0.03
    net.config[nengo.Ensemble].max_rates = max_rates

    # references
    net.inputs = inputs
    net.seed = seed
    net.synapse = synapse
    net.nNeurons = nNeurons

    net.ramp = ramp
    net.threshold = threshold
    net.relative = relative

    func_input = lambda t: net.inputs.sample(t)
    func_threshold = lambda t: net.threshold
    func_ramp = lambda x: net.synapse * net.ramp * x
    func_value = lambda x: [x[0]-x[1]*net.relative, x[1]-x[0]*net.relative]  # raw evidence vs relative advantage

    ePos = nengo.dists.Choice([[1]])
    iPos = nengo.dists.Uniform(0, 1)

    with net:
        # Inputs
        environment = nengo.Node(func_input)
        thr_input = nengo.Node(func_threshold)
        # Ensembles
        perception = nengo.Ensemble(nNeurons, nActions)
        accumulator = nengo.Ensemble(nNeurons, nActions, radius=rA)
        value = nengo.Ensemble(nNeurons, nActions, radius=net.threshold)
        thresh = nengo.Ensemble(nNeurons, 1, radius=net.threshold)
        gate = nengo.Ensemble(nNeurons, 1, encoders=ePos, intercepts=iPos)
        action = nengo.networks.EnsembleArray(nNeurons, nActions, encoders=ePos, intercepts=iPos)
        # Connections
        nengo.Connection(environment, perception)  # external inputs
        nengo.Connection(perception, accumulator, synapse=net.synapse, function=func_ramp)  # send percepts to accumulator
        # nengo.Connection(accumulator, accumulator, synapse=net.synapse) # recurrent connection for accumulation
        nengo.Connection(accumulator, value, function=func_value)  # compute value from evidence in accumulator
        nengo.Connection(value, action.input)
        nengo.Connection(thr_input, thresh)  # external inputs
        # nengo.Connection(thresh, gate)
        nengo.Connection(gate, action.input, transform=[[-1],[-1]], seed=seed)  # inhibition via decision criteria
        if w_or_d=='save':
            conn_accumulator = nengo.Connection(accumulator, accumulator, synapse=net.synapse, seed=seed) # recurrent cortical connection for accumulation    
            conn_threshold = nengo.Connection(thresh, gate, seed=seed)  # corticostriatal white matter
        elif w_or_d=="d":
            conn_accumulator = nengo.Connection(accumulator.neurons, accumulator, synapse=net.synapse, transform=w_accumulator, seed=seed)
            conn_threshold = nengo.Connection(thresh.neurons, gate, transform=w_threshold, seed=seed)
        elif w_or_d=="w":
            conn_accumulator = nengo.Connection(accumulator.neurons, accumulator.neurons, synapse=net.synapse, transform=w_accumulator, seed=seed)
            conn_threshold = nengo.Connection(thresh.neurons, gate.neurons, transform=w_threshold, seed=seed)
        # Probes
        net.pInputs = nengo.Probe(environment, synapse=None)
        net.pPerception = nengo.Probe(perception)
        net.pAccumulator = nengo.Probe(accumulator)
        net.pValue = nengo.Probe(value)
        net.pGate = nengo.Probe(gate)
        net.pAction = nengo.Probe(action.output)
        net.pSpikes = nengo.Probe(value.neurons, synapse=spike_filter)
        net.accumulator = accumulator
        net.gate = gate
        net.thresh = thresh
        net.value = value
        net.action = action
        net.conn_accumulator = conn_accumulator
        net.conn_threshold = conn_threshold

    return net

pid = sys.argv[1]
label = sys.argv[2]
trials = 100
emphases = ['speed', 'accuracy']
ages = ['young', 'old']

nNeurons = 500
rA = 1.0
max_rates = nengo.dists.Uniform(60, 80)
perception_seed = 0
dt = 0.001
tmin = 0.01
tmax = 2
amin = 0.15

with open(f"data/forstmann_optimized_params_{label}.json") as f:
    params = json.load(f)
param = params[pid]
ramp = param['ramp']
threshold_speed = param['threshold_speed']
threshold_accuracy = param['threshold_accuracy']
relative = param['relative']
dt_sample = param['dt_sample']
sigma = param['sigma']
coherence = param['coherence']


columns = ['type', 'pid', 'age', 'emphasis', 'trial', 'error', "RT"]
dfs = []
inputs = DotPerception(nActions=2, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
for e, emphasis in enumerate(emphases):
    # inputs = DotPerception(nActions=nActions, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
    inputs.create(coherence=coherence)
    if emphasis=='speed': threshold = threshold_speed
    if emphasis=='accuracy': threshold = threshold_accuracy
    # print(f"emphasis {emphasis}")
    for task_trial in range(trials):
        if 'young' in ages:
            net_young = build_network(inputs, w_or_d='save', nActions=2, nNeurons=nNeurons, rA=rA, seed=task_trial,
                                      max_rates=max_rates, ramp=ramp, threshold=threshold, relative=relative)
            # net_young = build_network(inputs, w_or_d='save', nActions=nActions, nNeurons=nNeurons, rA=rA, seed=task_trial,
            #                           max_rates=max_rates, ramp=ramp, threshold=threshold, relative=relative,
            #                           perception_noise=1e-10, accumulator_noise=1e-10)
            sim_young = nengo.Simulator(net_young, progress_bar=False)
            choice = None
            while choice==None:
                sim_young.run(dt)
                tnow = sim_young.trange()[-1]
                if np.any(sim_young.data[net_young.pAction][-1,:] > amin) and tnow>tmin:
                    choice = np.argmax(sim_young.data[net_young.pAction][-1,:])
                    RT = tnow
                if sim_young.trange()[-1] > tmax:
                    choice = np.argmax(sim_young.data[net_young.pValue][-1,:])
                    RT = tnow
            error = 0 if choice==inputs.correct else 100
            dfs.append(pd.DataFrame([['model', pid, 'young', emphasis, task_trial, error, RT]], columns=columns))

        if 'old' in ages:
            # effectively "age" the model from a functional, young model to an impaired "elderly" model
            w_acc, w_thr = age_weights(net_young, sim_young, w_or_d, method, degrade_accumulator, degrade_threshold, seed=perception_seed)
            net_old = build_network(inputs, w_or_d=w_or_d, w_accumulator=w_acc, w_threshold=w_thr,
                                    nActions=nActions, nNeurons=nNeurons, rA=rA, seed=task_trial,
                                    max_rates=max_rates, ramp=ramp, threshold=threshold, relative=relative)
            # net_old = build_network(inputs, w_or_d=w_or_d, w_accumulator=w_acc, w_threshold=w_thr,
            #                         nActions=nActions, nNeurons=nNeurons, rA=rA, seed=task_trial,
            #                         max_rates=max_rates, ramp=ramp, threshold=threshold, relative=relative,
            #                         perception_noise=perception_noise, accumulator_noise=accumulator_noise)
            sim_old = nengo.Simulator(net_old, progress_bar=False)
            choice = None
            while choice==None:
                sim_old.run(dt)
                tnow = sim_old.trange()[-1]
                if np.any(sim_old.data[net_old.pAction][-1,:] > amin) and tnow>tmin:
                    choice = np.argmax(sim_old.data[net_old.pAction][-1,:])
                    RT = tnow
                if sim_old.trange()[-1] > tmax:
                    choice = np.argmax(sim_old.data[net_old.pValue][-1,:])
                    RT = tnow
                    error = 0 if choice==inputs.correct else 100
            dfs.append(pd.DataFrame([['model', pid, 'old', emphasis, task_trial, error, RT]], columns=columns))

data = pd.concat(dfs, ignore_index=True)
data.to_pickle(f"data/forstmann_rerun_{pid}_{label}.pkl")