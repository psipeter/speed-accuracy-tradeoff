import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import optuna
import json
import sys
from scipy.stats import gaussian_kde

def run_DD(NDT, R, S, T, V, tmax, dt, rng):
    max_samples = int(tmax/dt)
    zeros = np.zeros((int(NDT/dt)))
    drift_samples = rng.normal(R, V, size=max_samples)
    samples = np.hstack([zeros, S, drift_samples])
    dv = np.cumsum(samples)[:max_samples]
    if np.any(np.abs(dv) >= T):
        RT = np.argwhere(np.abs(dv) >= T)[0][0]
        accuracy = 100 if dv[RT] >= T else 0
    else:
        RT = max_samples
        accuracy = 100 if rng.uniform(0,1)<0.5 else 0
    return accuracy, RT, dv

def get_loss(simulated, empirical, dPs, max_cues):
    total_loss = 0
    eval_points = np.linspace(0, max_cues, 1000)
    for dP in dPs:
        subsim = simulated.query("dP==@dP")
        subemp = empirical.query("dP==@dP")
        rts_sim = subsim['cues'].to_numpy()
        rts_emp = subemp['cues'].to_numpy()
        unique_sim = subsim['cues'].unique()
        unique_emp = subemp['cues'].unique()
        if len(unique_sim)==1 or len(unique_emp)==1:
            # if RTs have zero variance, use a mean error calculation instead
            mean_emp = np.mean(rts_emp)
            mean_sim = np.mean(rts_sim)
            mean_loss = 10*np.abs(mean_emp - mean_sim)
            total_loss += mean_loss
            # print('mean', mean_loss)
        else:
            # compute the RMS error between the evaluated KDEs
            kde_emp = gaussian_kde(rts_emp, bw_method='scott')
            kde_sim = gaussian_kde(rts_sim, bw_method='scott')
            estimate_emp = kde_emp.evaluate(eval_points)
            estimate_sim = kde_sim.evaluate(eval_points)
            estimate_emp = estimate_emp / np.sum(estimate_emp)
            estimate_sim = estimate_sim / np.sum(estimate_sim)
            kde_loss = 1000*np.sqrt(np.mean(np.square(estimate_emp - estimate_sim)))
            total_loss += kde_loss
            # print('kde', kde_loss)
    return total_loss


def objective(trial, pid, dPs=[0.2], experiment_time=4000, max_cues=12, dt=0.01, t_cue=1, rerun=False, params=None):
    empirical = pd.read_pickle("data/fiedler_trial.pkl").query("max_cues==@max_cues & id==@pid")
    if not rerun:
        T = trial.suggest_float("T", 0.1, 10, step=0.1) # decision threshold for speed emphasis
        # mu_nd = trial.suggest_float("mu_nd", 0, 2*max_cues, step=1)  # mean of non-decision time distribution
        mu_nd = trial.suggest_categorical("mu_nd", [0])  # mean of non-decision time distribution
        sigma_nd = trial.suggest_categorical("sigma_nd", [0])  # zero variance of non-decision time distribution
        mu_r0 = trial.suggest_float("mu_r0", 0.001, 0.1, step=0.001)  # R = mu_r0 * dP    
        sigma_r0 = trial.suggest_float("sigma_r0", 0.001, 0.1, step=0.001)
        mu_s = trial.suggest_categorical("mu_s", [0]) # mean of starting point distribution across trials is zero
        # sigma_s = trial.suggest_float("sigma_s", 0.01, 0.3, step=0.01) # no variance in starting point
        sigma_s = trial.suggest_categorical("sigma_s", [0]) # no variance in starting point
        V = trial.suggest_float("V", 0.01, 0.3, step=0.01) # drift variability
    else:
        T = params['T']
        mu_nd = params['mu_nd']
        sigma_nd = params['sigma_nd']
        mu_r0 = params['mu_r0']
        sigma_r0 = params['sigma_r0']
        mu_s = params['mu_s']
        sigma_s = params['sigma_s']
        V = params['V']

    perception_seed = 0
    network_seed = 0
    rng = np.random.RandomState(seed=network_seed)

    columns = ['type', 'dP', 'trial', 'cues', 'accuracy', 'id', 'max_cues']
    dfs = []
    for dP in dPs:
        total_time = 0
        task_trial = 0
        while total_time < experiment_time:
            mu_r = rng.normal(mu_r0, sigma_r0)
            NDT = rng.normal(mu_nd, sigma_nd)
            S = rng.normal(mu_s, sigma_s)
            R = mu_r * dP
            accuracy, RT, dv = run_DD(NDT, R, S, T, V, 2*max_cues, dt, rng)
            cues = np.ceil((RT*dt)/t_cue)
            # print(RT, RT*dt, cues, accuracy, total_time)
            dfs.append(pd.DataFrame([['DD', dP, task_trial, cues, accuracy, pid, max_cues]], columns=columns))
            total_time += RT*dt
            task_trial += 1

    simulated = pd.concat(dfs, ignore_index=True)
    loss = get_loss(simulated, empirical, dPs, 2*max_cues)
    if rerun:
        simulated.to_pickle(f"data/fiedler_rerun_DD_{pid}.pkl")
    return loss

if __name__ == '__main__':

    pid = int(sys.argv[1])
    optuna_trials = 1000

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study_name = f"fiedler_DD_{pid}"
    study = optuna.create_study(
        study_name=study_name,
        # storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, pid), n_trials=optuna_trials)
    with open(f"data/fiedler_params_DD_{pid}.json", 'w') as f:
        json.dump(study.best_params, f, indent=1)  # save parameters
    objective(None, pid, dPs=[0.4, 0.2, 0.1], rerun=True, params=study.best_params)  # rerun and save data