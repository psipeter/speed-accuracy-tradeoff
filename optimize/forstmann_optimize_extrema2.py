import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
# import mysql.connector
import sys
from model import DotPerception, detect_extrema_dot_motion

def chi_squared_distance(a,b):
    distance = 0
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    # print(a, b, distance)
    return distance

def get_loss(simulated, empirical, emphases):
    total_loss = 0
    bins = np.arange(0.0, 1.5, 0.1)
    for emphasis in emphases:
        for age in ages:
            rts_sim = simulated.query("emphasis==@emphasis")['RT'].to_numpy()
            rts_emp = empirical.query("emphasis==@emphasis")['RT'].to_numpy()
            err_sim = simulated.query("emphasis==@emphasis")['error'].mean()
            err_emp = empirical.query("emphasis==@emphasis")['error'].mean()
            hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
            hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
            normed_hist_rts_sim = hist_rts_sim / len(rts_sim)
            normed_hist_rts_emp = hist_rts_emp / len(rts_emp)
            chi_loss = chi_squared_distance(normed_hist_rts_sim, normed_hist_rts_emp)
            mean_loss = np.abs(np.mean(rts_sim) - np.mean(rts_emp))
            median_loss = np.abs(np.median(rts_sim) - np.median(rts_emp))
            err_loss = np.abs(err_sim - err_emp)
            total_loss += chi_loss
            total_loss += mean_loss
            total_loss += err_loss
    return total_loss

def objective(trial, pid, age):
    # let optuna choose the next parameters
    emphases = ['speed', 'neutral', 'accuracy']

    task_trials = 1000
    threshold1 = trial.suggest_float("threshold1", 0.1, 3.0, step=0.001)
    threshold2 = trial.suggest_float("threshold2", 0.1, 3.0, step=0.001)
    threshold3 = trial.suggest_float("threshold3", 0.1, 3.0, step=0.001)
    dt_sample = trial.suggest_float("dt_sample", 0.001, 0.1, step=0.001)
    sigma = trial.suggest_float("sigma", 0.01, 0.6, step=0.01)
    nd_mean = trial.suggest_float("nd_mean", 0.01, 2.0, step=0.001)  # mean of non-decision time distribution
    nd_sigma = trial.suggest_float("nd_sigma", 0.01, 0.6, step=0.001)  # variance of non-decision time distribution

    coherence = 0.1
    nActions = 2
    perception_seed = 0
    network_seed = 0
    dt = 0.001
    tmax = 1.5
    thresholds = [threshold1, threshold2, threshold3]
    rng = np.random.RandomState(seed=network_seed)
    # print(f"threshold {threshold}, sigma {sigma}, dt_sample {dt_sample}")

    # run task_trials iterations of the task, measuring simulated reaction times and accuracies
    columns = ['type', 'pid', 'age', 'emphasis','trial', 'error', "RT"]
    dfs = []
    for e, emphasis in enumerate(emphases):
        inputs = DotPerception(nActions=2, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
        threshold = thresholds[e]
        for t in range(task_trials):
            inputs.create(coherence=coherence)
            choice, RT = detect_extrema_dot_motion(inputs, threshold,
                tiebreaker="random", tmax=tmax, seed=network_seed)
            ndt = rng.normal(nd_mean, nd_sigma)
            ndt = np.max([0, ndt])
            RT += ndt  # add non-decision time to model's RT
            RT = np.min([tmax, RT])
            error = 0 if choice==inputs.correct else 100
            dfs.append(pd.DataFrame([['extrema', pid, age, emphasis, t, error, RT]], columns=columns))
            # print(f"emphasis {emphasis}, coherence {coherence}, trial {t}, RT {RT}")
    simulated = pd.concat(dfs, ignore_index=True)
    empirical = pd.read_pickle("data/forstmann2011.pkl").query("age==@age")
    loss = get_loss(simulated, empirical, emphases)
    return loss


if __name__ == '__main__':

    pid = sys.argv[1]
    age = sys.argv[2]
    label = sys.argv[3]
    study_name = f"forstmann_{pid}_{age}_{label}"
    optuna_trials = 10000

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
    study.optimize(lambda trial: objective(trial), n_trials=optuna_trials)