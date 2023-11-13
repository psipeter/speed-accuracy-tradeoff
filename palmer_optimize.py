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
        bins = np.arange(0.2, 1.0, 0.1)
    if condition=="normal":
        bins = np.arange(0.2, 2.0, 0.1)
    if condition=="accuracy":
        bins = np.arange(0.2, 4.0, 0.2)
    loss = 0
    for coherence in coherences:
        coh = 100*coherence
        rts_sim = simulated.query("coherence==@coherence")['RT'].to_numpy()
        rts_emp = empirical.query("coherence==@coh")['RT'].to_numpy()
        # accs_sim = simulated.query("coherence==@coherence")['accuracy']
        # accs_emp = empirical.query("coherence==@coherence")['accuracy']
        hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
        hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
        loss += chi_squared_distance(hist_rts_sim, hist_rts_emp)
    return loss

def RT_trial(net, nActions, dt=0.001, progress_bar=False, tmax=10):
    sim = nengo.Simulator(net, progress_bar=False)
    choice = None
    RT = None
    while choice==None:
        sim.run(dt)
        if np.any(sim.data[net.pAction][-1,:] > 0):
            choice = np.argmax(sim.data[net.pAction][-1,:])
            RT = sim.trange()[-1]
        if sim.trange()[-1] > tmax:
            choice = np.argmax(sim.data[net.pValue][-1,:])            
            RT = sim.trange()[-1]
    correct = 1 if choice==net.inputs.correct else 0
    return 100*correct, RT

def objective(trial, name, condition):
    # let optuna choose the next parameters
    nActions = 2
    task_trials = 3
    perception_seed = 0
    coherences = [0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
    ramp = trial.suggest_float("ramp", 0.5, 2.0, step=0.01)
    threshold = trial.suggest_float("threshold", 0.2, 0.8, step=0.01)
    empirical = pd.read_pickle("data/palmer2005.pkl").query(f"name==@name & condition==@condition")

    # run task_trials iterations of the task, measuring simulated reaction times and accuracies
    dfs = []
    inputs = DotPerception(nActions=nActions, seed=perception_seed)
    for c, coherence in enumerate(coherences):
        for t in range(task_trials):
            # print(f"coherence {coherence}, trial {t}")
            inputs.create(coherence=coherence)
            net = build_network(inputs, nActions=nActions, seed=t, ramp=ramp, threshold=threshold)
            acc, rt = RT_trial(net, nActions=nActions)
            print(acc, rt)
            dfs.append(pd.DataFrame([[coherence, t, rt, acc]], columns=('coherence', 'trial', 'RT', 'accuracy')))
    simulated = pd.concat(dfs, ignore_index=True)
    loss = get_loss(simulated, empirical, coherences, condition)
    print(f"loss {loss}")
    return loss


if __name__ == '__main__':

    name = sys.argv[1]
    condition = sys.argv[2]
    label = sys.argv[3]
    optuna_trials = 2

    # objective(None, name, condition)

    study = optuna.create_study(
          study_name=f"{name}_{condition}_{label}",
          load_if_exists=True,
          direction="minimize")
    study.optimize(lambda trial: objective(trial, name, condition), n_trials=optuna_trials)

    # host = "gra-dbaas1.computecanada.ca"
    # user = "psipeter"
    # password = ""
    # study = optuna.create_study(
    #     study_name=f"{name}_{condition}_{label}",
    #     storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
    #     load_if_exists=True,
    #     direction="minimize")
    # study.optimize(lambda trial: objective(trial, name, condition), n_trials=optuna_trials)
