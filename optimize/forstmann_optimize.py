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
        thresh = nengo.Ensemble(nNeurons, 1, radius=2*net.threshold)
        gate = nengo.Ensemble(nNeurons, 1, encoders=ePos, intercepts=iPos, radius=2*net.threshold)
        action = nengo.networks.EnsembleArray(nNeurons, nActions, encoders=ePos, intercepts=iPos)
        # Connections
        nengo.Connection(environment, perception)  # external inputs
        nengo.Connection(perception, accumulator, synapse=net.synapse, function=func_ramp)  # send percepts to accumulator
        # nengo.Connection(accumulator, accumulator, synapse=net.synapse) # recurrent connection for accumulation
        nengo.Connection(accumulator, value, function=func_value)  # compute value from evidence in accumulator
        nengo.Connection(value, action.input)
        nengo.Connection(thr_input, thresh)  # external inputs
        # nengo.Connection(thresh, gate)
        nengo.Connection(gate, action.ea_ensembles[0], transform=-1, seed=seed)  # inhibition via decision criteria
        nengo.Connection(gate, action.ea_ensembles[1], transform=-1, seed=seed)  # inhibition via decision criteria
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

def objective(trial, pid):

    trials = 100
    emphases = ['speed', 'accuracy']

    ramp = trial.suggest_float("ramp", 0.5, 2.0, step=0.001)
    threshold_speed = trial.suggest_float("threshold_speed", 0.01, 1.0, step=0.001)
    threshold_accuracy = trial.suggest_float("threshold_accuracy", 0.01, 1.0, step=0.001)
    relative = trial.suggest_float("relative", 0.01, 1.0, step=0.001)
    dt_sample = trial.suggest_float("dt_sample", 0.01, 0.1, step=0.001)
    sigma = trial.suggest_float("sigma", 0.01, 0.6, step=0.001)
    coherence = trial.suggest_categorical("coherence", [0.10])
    # coherence = trial.suggest_float("coherence", 0.01, 0.5, step=0.01)

    nNeurons = 70 # trial.suggest_categorical("nNeurons", [500])
    rA = 1.0  # trial.suggest_categorical("radius", [1.0])
    max_rates = nengo.dists.Uniform(60, 80)
    perception_seed = 0
    dt = 0.001
    tmin = 0.1
    tmax = 2
    
    columns = ['type', 'pid', 'age', 'emphasis', 'trial', 'error', "RT"]
    dfs = []
    for e, emphasis in enumerate(emphases):
        inputs = DotPerception(nActions=2, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
        inputs.create(coherence=coherence)
        if emphasis=='speed': threshold = threshold_speed
        if emphasis=='accuracy': threshold = threshold_accuracy
        print(f"emphasis {emphasis}")
        for task_trial in range(trials):
            net_young = build_network(inputs, w_or_d='save', nActions=2, nNeurons=nNeurons, rA=rA, seed=task_trial,
                                      max_rates=max_rates, ramp=ramp, threshold=threshold, relative=relative)
            sim_young = nengo.Simulator(net_young, progress_bar=False)
            choice = None
            while choice==None:
                sim_young.run(dt)
                tnow = sim_young.trange()[-1]
                if np.any(sim_young.data[net_young.pAction][-1,:] > 0.01) and tnow>tmin:
                    choice = np.argmax(sim_young.data[net_young.pAction][-1,:])
                    RT = tnow
                if sim_young.trange()[-1] > tmax:
                    choice = np.argmax(sim_young.data[net_young.pValue][-1,:])
                    RT = tnow
            error = 0 if choice==inputs.correct else 100
            dfs.append(pd.DataFrame([['model', pid, 'young', emphasis, task_trial, error, RT]], columns=columns))
    
    simulated = pd.concat(dfs, ignore_index=True)
    empirical = pd.read_pickle("data/forstmann2011.pkl").query("pid==@pid")
    # loss = get_loss(simulated, empirical, emphases)
    loss = get_kde_loss(simulated, empirical, emphases)
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