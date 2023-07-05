import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector
from model import Inputs, build_network, run_once

def chi_squared_distance(a,b):
    distance = 0
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    return distance

def get_loss(simulated, empirical):
    # return np.abs(simulated - empirical)
    print('simulated', simulated)
    print('empirical', empirical)
#     loss = np.sum(np.abs(simulated - empirical))
#     loss = 1 - entropy(simulated, empirical)
    loss = chi_squared_distance(simulated, empirical)
    print('loss', loss)
    return loss

def objective(trial, pid):
    # let optuna choose the next parameters
    seed = trial.suggest_categorical("seed", range(100))
    T = trial.suggest_float("T", 0.0, 1.0, step=0.01)
    m = trial.suggest_float("m", 0.0, 1.0, step=0.01)
    # tau = trial.suggest_float("tau", 0.0, 0.3, step=0.01)
    # delta = trial.suggest_float("delta", 0.0, 1.0, step=0.01)

    tau = 0
    delta = 0
    # each participant has a participant ID (a long string) and an associated "pid" (integer) in the database
    # load this data to compute the loss between simulated and model behavior
    with open('data/pids.pkl', 'rb') as f:
        pids = pickle.load(f)
    participant_ID = pids[pid]
    empirical = pd.read_pickle("data/empirical.pkl").query("maxSamples==12 & delta==0.1 & participant_id==@participant_ID")
    empirical_hist = np.histogram(empirical['cues'].to_numpy(), bins=np.arange(0, 27, 3))[0]

    # for each trial performed by the participant, run the model
    simulated_cues = []
    # total_loss = 0
    inputs = Inputs(deltaP=0.1, maxCues=12, seed=pid, empirical=empirical)
    net = build_network(inputs, seed=seed, T=T, m=m, tau=tau, delta=delta)
    task_trials = empirical.shape[0]
    for task_trial in range(task_trials):
        net.inputs.set_AB(trial=task_trial)
        is_correct, cues_sampled = run_once(net)
        simulated_cues.append(cues_sampled)
        # loss = get_loss(cues_sampled, empirical['cues'].to_numpy()[trial])
        # total_loss += loss

    simulated_hist = np.histogram(simulated_cues, bins=np.arange(0, 27, 3))[0]
    total_loss = get_loss(simulated_hist, empirical_hist)
    return total_loss


if __name__ == '__main__':

    pid = int(sys.argv[1])
    study_name = sys.argv[2]
    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = ""
    n_trials = 200

    study = optuna.create_study(
          study_name=f"{study_name}",
          storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
          load_if_exists=True,
          direction="minimize")
    study.optimize(lambda trial: objective(trial, pid), n_trials=n_trials)