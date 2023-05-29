import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import logging
import sys
import mysql.connector
from model import Inputs, build_network, run_once


def get_loss(simulated_cues, empirical_cues, trial):
    delta_cues = np.abs(simulated_cues - empirical_cues[trial])
    return delta_cues

def objective(trial):
    # let optuna choose the next parameters
    pid = 1  # tria.suggest_int('pid', 1)
    seed = pid
    T = trial.suggest_float("T", 0.1, 0.5, step=0.01)
    m = trial.suggest_float("m", 0.1, 0.3, step=0.01)
    tau = trial.suggest_float("tau", 0.01, 0.1, step=0.01)
    delta = trial.suggest_float("delta", 0.1, 1.0, step=0.1)

    # each participant has a participant ID (a long string) and an associated "pid" (integer) in the database
    # load this data to compute the loss between simulated and model behavior
    with open('data/pids.pkl', 'rb') as f:
        pids = pickle.load(f)
    participant_ID = pids[pid]
    empirical = pd.read_pickle("data/empirical.pkl").query("maxSamples==12 & delta==0.1 & participant_id==@participant_ID")

    # for each trial performed by the participant, run the model
    total_loss = 0
    inputs = Inputs(deltaP=0.1, maxCues=12, seed=seed, empirical=empirical)
    net = build_network(inputs, seed=seed, T=T, m=m, tau=tau, delta=delta)
    for task_trial in range(empirical.shape[0]):
        net.inputs.set_AB(trial=task_trial)
        is_correct, cues_sampled = run_once(net)
        loss = get_loss(cues_sampled, empirical['cues'].to_numpy(), task_trial)
        print(f"trial {task_trial}, loss={loss}")
        total_loss += loss
        # uped by optuna to prune hyperparameter trials that already give bad results part way through the task
        trial.report(total_loss, task_trial)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return total_loss


if __name__ == '__main__':

    mydb = mysql.connector.connect(host="localhost", user="root", password="0pHkVsqvDQ1E0OHk3Vhw")
    mycursor = mydb.cursor()
    # mycursor.execute("CREATE DATABASE sql_test")
    mycursor.execute("SHOW DATABASES")
    for x in mycursor:
        print(x)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name="test", storage="sqlite:///test.db", load_if_exists=True,
        direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20, timeout=1000)

    best_params = study.best_params
    best_loss = study.best_value

    T = best_params["T"]
    m = best_params["m"]
    tau = best_params["tau"]
    delta = best_params["delta"]

    # print(f"loss={best_loss}, T={T}, m={m}, tau={tau}, delta={delta}")
    # np.savez(f"data/fit_params.npz", seed=seed, T=T, m=m, tau=tau, delta=delta)