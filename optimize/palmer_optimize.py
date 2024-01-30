import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector
from network_revised import DotPerception, build_network

def chi_squared_distance(a,b):
    distance = 0
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    # print(a, b, distance)
    return distance

def get_loss(simulated, empirical, coherences, condition):
    total_loss = 0
    if condition=="speed":
        bins = np.arange(0.0, 1.1, 0.1)
    if condition=="normal":
        bins = np.arange(0.0, 2.3, 0.3)
    if condition=="accuracy":
        bins = np.arange(0.0, 4.5, 0.5)
    for coherence in coherences:
        coh = 100*coherence
        rts_sim = simulated.query("coherence==@coherence")['RT'].to_numpy()
        rts_emp = empirical.query("coherence==@coh")['RT'].to_numpy()
        hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
        hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
        normed_hist_rts_sim = hist_rts_sim / len(rts_sim)
        normed_hist_rts_emp = hist_rts_emp / len(rts_emp)
        chi_loss = chi_squared_distance(normed_hist_rts_sim, normed_hist_rts_emp)
        mean_loss = np.abs(np.mean(rts_sim) - np.mean(rts_emp))
        total_loss += chi_loss
        # total_loss += mean_loss
    return total_loss

def objective(trial, name, condition):
    # let optuna choose the next parameters
    dt = 0.001
    nActions = 2
    task_trials = 100
    nNeurons = 500
    dt_sample = 0.01
    sigma = 0.3
    rA = 4.0
    tmax = 3 if condition in ['speed', 'normal'] else 5
    coherences = [0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
    perception_seed = 0  # trial.suggest_categorical("perception_seed", range(1000))
    # network_seed = 0  # trial.suggest_categorical("network_seed", range(1000))
    ramp = trial.suggest_float("ramp", 0.5, 2.0, step=0.01)
    threshold = trial.suggest_float("threshold", 0.0, 1.0, step=0.01)
    relative = trial.suggest_float("relative", 0.0, 1.0, step=0.01)

    empirical = pd.read_pickle("data/palmer2005.pkl").query(f"name==@name & condition==@condition")
    print(f"ramp {ramp}, threshold {threshold}, relative {relative}")

    # run task_trials iterations of the task, measuring simulated reaction times and accuracies
    dfs = []
    inputs = DotPerception(nActions=nActions, dt_sample=dt_sample, seed=perception_seed)
    for coherence in coherences:
        # print(f"coherence {coherence}")
        for t in range(task_trials):
            print(f"coherence {coherence}, trial {t}")
            inputs.create(coherence=coherence)
            # net = build_network(inputs, nActions=nActions, seed=network_seed, ramp=ramp, threshold=threshold, nNeurons=nNeurons, relative=relative, rA=rA)
            net = build_network(inputs, nActions=nActions, seed=t, ramp=ramp, threshold=threshold, nNeurons=nNeurons, relative=relative, rA=rA)
            sim = nengo.Simulator(net, progress_bar=False)
            choice = None
            while choice==None:
                sim.run(dt)
                if np.any(sim.data[net.pAction][-1,:] > 0.01):
                    choice = np.argmax(sim.data[net.pAction][-1,:])
                    RT = sim.trange()[-1]
                if sim.trange()[-1] > tmax:
                    choice = np.argmax(sim.data[net.pValue][-1,:])
                    RT = sim.trange()[-1]
            acc = 100 if choice==inputs.correct else 0
            dfs.append(pd.DataFrame([[coherence, t, RT, acc]], columns=('coherence', 'trial', 'RT', 'accuracy')))
            # print(RT, acc)
    simulated = pd.concat(dfs, ignore_index=True)
    loss = get_loss(simulated, empirical, coherences, condition)
    return loss


if __name__ == '__main__':

    name = sys.argv[1]
    condition = sys.argv[2]
    label = sys.argv[3]
    study_name=f"{name}_{condition}_{label}"
    optuna_trials = 100

    # objective(None, name, condition)
    # raise

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, name, condition), n_trials=optuna_trials)