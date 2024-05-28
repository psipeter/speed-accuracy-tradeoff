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

def get_loss(simulated, empirical, emphases, ages):
    total_loss = 0
    bins = np.arange(0.0, 1.5, 0.1)
    for emphasis in emphases:
        for age in ages:
            rts_sim = simulated.query("emphasis==@emphasis & age==@age")['RT'].to_numpy()
            rts_emp = empirical.query("emphasis==@emphasis & age==@age")['RT'].to_numpy()
            hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
            hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
            normed_hist_rts_sim = hist_rts_sim / len(rts_sim)
            normed_hist_rts_emp = hist_rts_emp / len(rts_emp)
            chi_loss = chi_squared_distance(normed_hist_rts_sim, normed_hist_rts_emp)
            mean_loss = np.abs(np.mean(rts_sim) - np.mean(rts_emp))
            median_loss = np.abs(np.median(rts_sim) - np.median(rts_emp))
            total_loss += chi_loss
            total_loss += median_loss
            # total_loss += mean_loss
    return total_loss

def objective(trial):
    # let optuna choose the next parameters
    ages = ['young']  # ['young', 'old']
    emphases = ['accuracy', 'speed']

    task_trials = 300
    threshold1 = trial.suggest_float("threshold1", 0.1, 2.0, step=0.001)
    # threshold2 = trial.suggest_float("threshold2", 0.1, 2.0, step=0.001)
    threshold3 = trial.suggest_float("threshold3", 0.1, 2.0, step=0.001)
    dt_sample = trial.suggest_categorical("dt_sample", [0.01])
    sigma = trial.suggest_categorical("sigma", [0.3])
    nd_mean = trial.suggest_float("nd_mean", 0.01, 2.0, step=0.001)  # mean of non-decision time distribution
    nd_sigma = trial.suggest_float("nd_sigma", 0.01, 0.5, step=0.001)  # variance of non-decision time distribution

    coherence = 0.1
    nActions = 2
    perception_seed = 0
    network_seed = 0
    dt = 0.001
    tmax = 1.5
    thresholds = [threshold1, threshold3]
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
            dfs.append(pd.DataFrame([['extrema', t, 'young', emphasis, t, error, RT]], columns=columns))
            # print(f"emphasis {emphasis}, coherence {coherence}, trial {t}, RT {RT}")
    simulated = pd.concat(dfs, ignore_index=True)
    empirical = pd.read_pickle("data/forstmann2011.pkl")
    loss = get_loss(simulated, empirical, emphases, ages)
    return loss


if __name__ == '__main__':

    label = sys.argv[1]
    study_name = f"forstmann_{label}"
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