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
from model import DotPerception

def build_network(inputs, w_accumulator, w_speed, nActions=2, nNeurons=500, synapse=0.1, seed=0,
    ramp=1, threshold=0.5, speed=-0.1, relative=0,
    max_rates=nengo.dists.Uniform(60, 80), rA=1, save_w=False, weights_or_decoders="decoders"):
    
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
        boundary = nengo.Ensemble(nNeurons, 1, seed=seed)
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

def objective(trial, pid):

    emphases = ['speed', 'neutral', 'accuracy']

    ramp = trial.suggest_float("ramp", 0.5, 2.0, step=0.01)
    threshold = trial.suggest_float("threshold", 0.01, 1.0, step=0.01)
    relative = trial.suggest_float("relative", 0.01, 1.0, step=0.01)
    speed = trial.suggest_float("speed", -0.2, -0.01, step=0.01)
    e1 = 0.0  # trial.suggest_float("e1", 0.01, 1.0, step=0.01)
    e2 = trial.suggest_float("e2", 0.01, 1.0, step=0.01)
    e3 = 1.0  # trial.suggest_float("e3", 0.01, 1.0, step=0.01)
    # dt_sample = trial.suggest_float("dt_sample", 0.001, 0.1, step=0.001)
    dt_sample = trial.suggest_categorical("dt_sample", [0.1], step=0.001)
    # sigma = trial.suggest_float("sigma", 0.01, 0.7, step=0.01)
    sigma = trial.suggest_categorical("sigma", [0.2])
    coherence = trial.suggest_categorical("coherence", [0.1])

    degrade_accumulator = 0
    degrade_speed = 0
    nNeurons = 500 # trial.suggest_categorical("nNeurons", [500])
    rA = 1.0  # trial.suggest_categorical("radius", [1.0])
    minRate = 60  # trial.suggest_categorical("minRate", [60])
    maxRate = 80 # trial.suggest_categorical("maxRate", [80])
    max_rates = nengo.dists.Uniform(minRate, maxRate)
    emphases_weighting = [e1, e2, e3]

    trials = 300

    weights_or_decoders = "decoders"
    nActions = 2
    perception_seed = 0
    dt = 0.001
    tmax = 1.5
    
    columns = ['type', 'pid', 'age', 'emphasis','trial', 'error', "RT"]
    dfs = []
    for e, emphasis in enumerate(emphases):
        inputs = DotPerception(nActions=nActions, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
        inputs.create(coherence=coherence)
        S = emphases_weighting[e] * speed
        for trial in range(trials):
            net_young = build_network(inputs, None, None, nActions=nActions, nNeurons=nNeurons, rA=rA, seed=trial,
                                      max_rates=max_rates, ramp=ramp, threshold=threshold, speed=S, relative=relative,
                                      save_w=True, weights_or_decoders=weights_or_decoders)
            sim_young = nengo.Simulator(net_young, progress_bar=False)
            # simulate the "young" network
            choice = None
            while choice==None:
                sim_young.run(dt)
                if np.any(sim_young.data[net_young.pAction][-1,:] > 0.01):
                    choice = np.argmax(sim_young.data[net_young.pAction][-1,:])
                    RT = sim_young.trange()[-1]
                if sim_young.trange()[-1] > tmax:
                    choice = np.argmax(sim_young.data[net_young.pValue][-1,:])
                    RT = sim_young.trange()[-1]
            error = 0 if choice==inputs.correct else 100
            dfs.append(pd.DataFrame([['model', trial, 'young', emphasis, trial, error, RT]], columns=columns))
    
    simulated = pd.concat(dfs, ignore_index=True)
    empirical = pd.read_pickle("data/forstmann2011.pkl").query("pid==@pid")
    loss = get_loss(simulated, empirical, emphases, ages)
    return loss

if __name__ == '__main__':

    pid = sys.argv[1]
    label = sys.argv[2]
    study_name = f"forstmann_{pid}_{label}"
    optuna_trials = 1000

    # objective(None)
    # raise

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, pid), n_trials=optuna_trials)