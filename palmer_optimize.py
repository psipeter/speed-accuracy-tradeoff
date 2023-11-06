import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector
from network_revised import Inputs_TwoDots, build_network, single_trial

def chi_squared_distance(a,b):
    distance = 0
    a = a / np.sum(a)
    b = b / np.sum(b)
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    # print(a, b, distance)
    return distance

def get_loss(simulated, empirical, coherences, condition):
    if condition=="speed":
        bins = np.arange(200, 1000, 100)
    if condition=="normal":
        bins = np.arange(200, 2000, 100)
    if condition=="accuracy":
        bins = np.arange(200, 4000, 200)
    loss = 0
    for coherence in coherences:
        coh = 100*coherence
        rts_sim = simulated.query("coherence==@coherence")['RT']
        rts_emp = empirical.query("coherence==@coh")['RT']
        # accs_sim = simulated.query("coherence==@coherence")['accuracy']
        # accs_emp = empirical.query("coherence==@coherence")['accuracy']
        hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
        hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
        loss += chi_squared_distance(hist_rts_sim, hist_rts_emp)
    return loss

def objective(trial, name, condition):
    # let optuna choose the next parameters
    task_trials = 10
    coherences = [0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
    ramp = trial.suggest_float("ramp", 0.5, 2.0, step=0.01)
    threshold = trial.suggest_float("threshold", 0.2, 0.8, step=0.01)
    empirical = pd.read_pickle("data/palmer2005.pkl").query(f"name==@name & condition==@condition")

    # run task_trials iterations of the task, measuring simulated reaction times and accuracies
    dfs = []
    inputs = Inputs_TwoDots(seed=0)
    for c, coherence in enumerate(coherences):
        for t in range(task_trials):
            # print(f"coherence {coherence}, trial {t}")
            inputs.create(coherence=coherence)
            net = build_network(inputs, seed=t, ramp=ramp, threshold=threshold)
            acc, rt = single_trial(net)
            dfs.append(pd.DataFrame([[coherence, t, rt, acc]], columns=('coherence', 'trial', 'RT', 'accuracy')))
    simulated = pd.concat(dfs, ignore_index=True)
    loss = get_loss(simulated, empirical, coherences, condition)
    # print(f"loss {loss}")
    return loss


if __name__ == '__main__':

    name = sys.argv[1]
    condition = sys.argv[2]
    label = sys.argv[3]
    optuna_trials = 1

    # objective(None, name, condition)

    # study = optuna.create_study(
    #       study_name=f"{name}_{condition}_{label}",
    #       load_if_exists=True,
    #       direction="minimize")
    # study.optimize(lambda trial: objective(trial, name, condition), n_trials=optuna_trials)

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = ""

    study = optuna.create_study(
        study_name=f"{name}_{condition}_{label}",
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, name, condition), n_trials=optuna_trials)
