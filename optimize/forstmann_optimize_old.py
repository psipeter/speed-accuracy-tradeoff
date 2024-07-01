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
import json
from scipy.stats import gaussian_kde
from model import DotPerception


def build_network(inputs, w_accumulator, w_speed, nActions=2, nNeurons=500, synapse=0.1, seed=0,
    ramp=1, threshold=0.5, speed=-0.1, relative=0,
    max_rates=nengo.dists.Uniform(60, 80), rA=1, rB=0.3, save_w=False, weights_or_decoders="decoders"):
    
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
    net.speed = speed
    net.relative = relative

    func_input = lambda t: net.inputs.sample(t)
    func_threshold = lambda t: net.threshold
    func_speed = lambda t: net.speed
    func_ramp = lambda x: net.synapse * net.ramp * x
    func_value = lambda x: [x[0]-x[1]*net.relative, x[1]-x[0]*net.relative]  # raw evidence vs relative advantage

    ePos = nengo.dists.Choice([[1]])
    iPos = nengo.dists.Uniform(0, 1)

    with net:
        # Inputs
        environment = nengo.Node(func_input)
        baseline_threshold = nengo.Node(func_threshold)
        speed_control = nengo.Node(func_speed)
        # Ensembles
        perception = nengo.Ensemble(nNeurons, nActions, seed=seed)
        accumulator = nengo.Ensemble(nNeurons, nActions, radius=rA, seed=seed)
        value = nengo.Ensemble(nNeurons, nActions, radius=net.threshold, seed=seed)
        boundary = nengo.Ensemble(nNeurons, 1, seed=seed, radius=rB)
        gate = nengo.Ensemble(nNeurons, 1, encoders=ePos, intercepts=iPos, seed=seed)
        action = nengo.networks.EnsembleArray(nNeurons, nActions, encoders=ePos, intercepts=iPos, seed=seed)
        # Connections
        nengo.Connection(environment, perception, seed=seed)  # external inputs
        nengo.Connection(perception, accumulator, synapse=net.synapse, function=func_ramp, seed=seed)  # send percepts to accumulator
        # recurrent cortical connection for accumulation
        if save_w:
            conn_accumulator = nengo.Connection(accumulator, accumulator, synapse=net.synapse, seed=seed)            
        elif weights_or_decoders=="decoders":
            conn_accumulator = nengo.Connection(accumulator.neurons, accumulator, synapse=net.synapse, transform=w_accumulator, seed=seed)
        elif weights_or_decoders=="weights":
            conn_accumulator = nengo.Connection(accumulator.neurons, accumulator.neurons, synapse=net.synapse, transform=w_accumulator, seed=seed)
        nengo.Connection(accumulator, value, function=func_value, seed=seed)  # compute value from evidence in accumulator
        nengo.Connection(value, action.input, seed=seed)
        nengo.Connection(baseline_threshold, gate, seed=seed)  # baseline activity for gate population
        nengo.Connection(speed_control, boundary, seed=seed)  # external inputs (from "cortex") sets decision threshold based on task instructions
        # corticostriatal white matter connection
        if save_w:
            conn_speed = nengo.Connection(boundary, gate, seed=seed)
        elif weights_or_decoders=="decoders":
            conn_speed = nengo.Connection(boundary.neurons, gate, transform=w_speed, seed=seed)
        elif weights_or_decoders=="weights":
            conn_speed = nengo.Connection(boundary.neurons, gate.neurons, transform=w_speed, seed=seed)
        nengo.Connection(gate, action.input, transform=-1*np.ones((nActions, 1)))  # inhibition via decision criteria
        # Probes
        net.pInputs = nengo.Probe(environment, synapse=None)
        net.pPerception = nengo.Probe(perception)
        net.pAccumulator = nengo.Probe(accumulator)
        net.pValue = nengo.Probe(value)
        net.pGate = nengo.Probe(gate)
        net.pAction = nengo.Probe(action.output)
        net.accumulator = accumulator
        net.value = value
        net.boundary = boundary
        net.gate = gate
        net.conn_accumulator = conn_accumulator
        net.conn_speed = conn_speed
    return net


def degrade_weights(net_young, pre_sim, degrade_accumulator, degrade_speed, seed=0, weights_or_decoders="weights"):
    young_accumulator = pre_sim.data[net_young.conn_accumulator].weights  # decoders
    young_speed = pre_sim.data[net_young.conn_speed].weights  # decoders
    old_accumulator = young_accumulator.copy()
    old_speed = young_speed.copy()
    rng = np.random.RandomState(seed=seed)

    if weights_or_decoders=="decoders":
        idx_accumulator = rng.choice(range(old_accumulator.shape[1]), size=int(degrade_accumulator*old_accumulator.shape[1]), replace=False)
        idx_speed = rng.choice(range(old_speed.shape[1]), size=int(degrade_speed*old_speed.shape[1]), replace=False)
        old_accumulator[:,idx_accumulator] = 0
        old_speed[:,idx_speed] = 0

    if weights_or_decoders=="weights":
        e_accumulator = pre_sim.data[net_young.accumulator].encoders
        e_speed = pre_sim.data[net_young.gate].encoders
        w_accumulator = e_accumulator @ young_accumulator
        w_speed = e_speed @ young_speed
        # print(e_accumulator.shape, old_accumulator.shape, w_accumulator.shape)
        flat_accumulator = w_accumulator.ravel().copy()
        flat_speed = w_speed.ravel().copy()
        idx_accumulator = rng.choice(range(flat_accumulator.shape[0]), size=int(degrade_accumulator*flat_accumulator.shape[0]), replace=False)
        idx_speed = rng.choice(range(flat_speed.shape[0]), size=int(degrade_speed*flat_speed.shape[0]), replace=False)
        flat_accumulator[idx_accumulator] = 0
        flat_speed[idx_speed] = 0
        old_accumulator = flat_accumulator.reshape(w_accumulator.shape)
        old_speed = flat_speed.reshape(w_speed.shape)
        # print(np.sum(w_accumulator), np.sum(old_accumulator))
        # print(np.sum(w_speed), np.sum(old_speed))

    return old_accumulator, old_speed


def chi_squared_distance(a,b):
    distance = 0
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    return distance

def get_loss(simulated, empirical, emphases):
    total_loss = 0
    bins = np.arange(0.0, 1.5, 0.1)
    for emphasis in emphases:
        rts_sim = simulated.query("emphasis==@emphasis")['RT'].to_numpy()
        rts_emp = empirical.query("emphasis==@emphasis")['RT'].to_numpy()
        hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
        hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
        normed_hist_rts_sim = hist_rts_sim / len(rts_sim)
        normed_hist_rts_emp = hist_rts_emp / len(rts_emp)
        chi_loss = chi_squared_distance(normed_hist_rts_sim, normed_hist_rts_emp)
        mean_loss = np.abs(np.mean(rts_sim) - np.mean(rts_emp))
        median_loss = np.abs(np.median(rts_sim) - np.median(rts_emp))
        total_loss += chi_loss
        # total_loss += median_loss
        # total_loss += mean_loss
    return total_loss

def get_kde_loss(simulated, empirical, emphases):
    total_loss = 0
    eval_points = np.linspace(0, 1.5, 1000)
    for emphasis in emphases:
        rts_sim = simulated.query("emphasis==@emphasis")['RT'].to_numpy()
        rts_emp = empirical.query("emphasis==@emphasis")['RT'].to_numpy()
        # check for zero variance and assign error accordingly
        if len(simulated.query("emphasis==@emphasis")['RT'].unique())==1:
            mean_emp = np.mean(rts_emp)
            mean_sim = np.mean(rts_sim)
            kde_loss = 10*np.abs(mean_emp - mean_sim)
            print('mean', kde_loss)
        else:     
            kde_emp = gaussian_kde(rts_emp, bw_method='scott')
            kde_sim = gaussian_kde(rts_sim, bw_method='scott')
            estimate_emp = kde_emp.evaluate(eval_points)
            estimate_sim = kde_sim.evaluate(eval_points)
            estimate_emp = estimate_emp / np.sum(estimate_emp)
            estimate_sim = estimate_sim / np.sum(estimate_sim)
            kde_loss = 1000*np.sqrt(np.mean(np.square(estimate_emp - estimate_sim)))
            print('kde', kde_loss)
        total_loss += kde_loss
        # error_sim = simulated.query("emphasis==@emphasis")['error'].mean()
        # error_emp = empirical.query("emphasis==@emphasis")['error'].mean()
        # error_loss = np.abs(error_sim - error_emp)
        # print('error', error_loss)
        # total_loss += error_loss
    return total_loss

def objective(trial):

    trials = 3
    emphases = ['speed', 'accuracy']

    degrade_speed = trial.suggest_float("degrade_speed", 0.0, 1.0, step=0.001)
    degrade_accumulator = trial.suggest_float("degrade_accumulator", 0.0, 1.0, step=0.001)

    nNeurons = 50 # trial.suggest_categorical("nNeurons", [500])
    rA = 1.0  # trial.suggest_categorical("radius", [1.0])
    rB = 0.3
    minRate = 60  # trial.suggest_categorical("minRate", [60])
    maxRate = 80 # trial.suggest_categorical("maxRate", [80])
    max_rates = nengo.dists.Uniform(minRate, maxRate)
    perception_seed = 0
    dt = 0.001
    tmax = 1.5
    emp = pd.read_pickle(f"data/forstmann2011.pkl").query("age=='young'")
    columns = ['type', 'pid', 'age', 'emphasis','trial', 'error', "RT"]
    dfs = []

    for pid in emp['pid'].unique():
        print(pid)
        with open(f"data/forstmann_optimized_params_june29.json") as f:
            params = json.load(f)[pid]
        coherence = params['coherence']
        dt_sample = params['dt_sample']
        ramp = params['ramp']
        relative = params['relative']
        sigma = params['sigma']
        speed = params['speed']
        threshold = params['threshold']        
        for e, emphasis in enumerate(emphases):
            inputs = DotPerception(nActions=2, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
            inputs.create(coherence=coherence)
            if emphasis=='speed': S=speed
            if emphasis=='accuracy': S=0
            for task_trial in range(trials):
                net_young = build_network(inputs, None, None, nActions=2,
                                          nNeurons=nNeurons, rA=rA, rB=rB, seed=task_trial,
                                          max_rates=max_rates, ramp=ramp, threshold=threshold, speed=S, relative=relative,
                                          save_w=True, weights_or_decoders="decoders")
                sim_young = nengo.Simulator(net_young, progress_bar=False)
                # age the model
                old_accumulator, old_speed = degrade_weights(net_young, sim_young, degrade_accumulator, degrade_speed, task_trial, "decoders")
                net_old = build_network(inputs, old_accumulator, old_speed,
                                        nActions=2, nNeurons=nNeurons, rA=rA, rB=rB, seed=task_trial,
                                        max_rates=max_rates, ramp=ramp, threshold=threshold, speed=S, relative=relative,
                                        save_w=False, weights_or_decoders="decoders")
                sim_old = nengo.Simulator(net_old, progress_bar=False)
                # simulate the "old" network
                choice = None
                while choice==None:
                    sim_old.run(dt)
                    if np.any(sim_old.data[net_old.pAction][-1,:] > 0.01):
                        choice = np.argmax(sim_old.data[net_old.pAction][-1,:])
                        RT = sim_old.trange()[-1]
                    if sim_old.trange()[-1] > tmax:
                        choice = np.argmax(sim_old.data[net_old.pValue][-1,:])
                        RT = sim_old.trange()[-1]
                error = 0 if choice==inputs.correct else 100
                dfs.append(pd.DataFrame([['model', pid, 'old', emphasis, task_trial, error, RT]], columns=columns))
    
    simulated = pd.concat(dfs, ignore_index=True)
    empirical = pd.read_pickle(f"data/forstmann2011.pkl").query("age=='old'")
    # loss = get_loss(simulated, empirical, emphases)
    loss = get_kde_loss(simulated, empirical, emphases)
    return loss

if __name__ == '__main__':

    label = sys.argv[1]
    study_name = f"forstmann_{label}"
    optuna_trials = 1000

    # objective(None)
    # raise

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study = optuna.create_study(
        study_name=study_name,
        # storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial), n_trials=optuna_trials)