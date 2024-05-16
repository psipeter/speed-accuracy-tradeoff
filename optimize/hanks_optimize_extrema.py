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

def get_loss(simulated, empirical, coherences, emphases):
    total_loss = 0
    bins = np.arange(0.0, 1.5, 0.1)
    for emphasis in emphases:
        for C in coherences:
            coh = 100*C
            rts_sim = simulated.query("emphasis==@emphasis & coherence==@C")['RT'].to_numpy()
            rts_emp = empirical.query("emphasis==@emphasis & coherence==@coh")['RT'].to_numpy()
            hist_rts_sim = np.histogram(rts_sim, bins=bins)[0]
            hist_rts_emp = np.histogram(rts_emp, bins=bins)[0]
            normed_hist_rts_sim = hist_rts_sim / len(rts_sim)
            normed_hist_rts_emp = hist_rts_emp / len(rts_emp)
            chi_loss = chi_squared_distance(normed_hist_rts_sim, normed_hist_rts_emp)
            mean_loss = np.abs(np.mean(rts_sim) - np.mean(rts_emp))
            median_loss = np.abs(np.median(rts_sim) - np.median(rts_emp))
            total_loss += chi_loss
            total_loss += median_loss
            # total_loss += 3*mean_loss
    return total_loss

def objective(trial):
    # let optuna choose the next parameters
    emphases = ["speed", "accuracy"]
    name = "E"
    task_trials = 1000
    threshold1 = trial.suggest_float("threshold1", 0.1, 2.0, step=0.01)
    threshold2 = trial.suggest_float("threshold2", 0.1, 2.0, step=0.01)
    dt_sample = trial.suggest_float("dt_sample", 0.001, 0.1, step=0.001)
    sigma = trial.suggest_float("sigma", 0.01, 0.7, step=0.01)
    tiebreaker = trial.suggest_categorical("tiebreaker", ['random'])  # makes no difference for RT, which determines loss
    nd_mean = trial.suggest_float("nd_mean", 0.01, 2.0, step=0.01)  # mean of non-decision time distribution
    nd_sigma = trial.suggest_float("nd_sigma", 0.01, 0.7, step=0.01)  # variance of non-decision time distribution
    tmax = 2
    coherences = [0.032, 0.064, 0.128, 0.256, 0.512]
    perception_seed = 0
    network_seed = 0
    rng = np.random.RandomState(seed=network_seed)
    # print(f"threshold {threshold}, sigma {sigma}, dt_sample {dt_sample}")

    # run task_trials iterations of the task, measuring simulated reaction times and accuracies
    columns = ['type', 'emphasis', 'coherence', 'trial', 'RT', 'accuracy']
    dfs = []
    for e, emphasis in enumerate(emphases):
        empirical = pd.read_pickle("data/hanks2014_behavior.pkl").query("id==@name")
        inputs = DotPerception(nActions=2, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
        threshold = thresholds[e]
        for coherence in coherences:
            for t in range(task_trials):
                inputs.create(coherence=coherence)
                choice, RT = detect_extrema_dot_motion(inputs, threshold, tiebreaker, tmax=tmax, seed=network_seed)
                ndt = rng.normal(nd_mean, nd_sigma)
                ndt = np.max([0, ndt])
                RT += ndt  # add non-decision time to model's RT
                RT = np.min([tmax, RT])
                acc = 100 if choice==inputs.correct else 0
                dfs.append(pd.DataFrame([['extrema', emphasis, coherence, t, RT, acc]], columns=columns))
                # print(f"emphasis {emphasis}, coherence {coherence}, trial {t}, RT {RT}")
    simulated = pd.concat(dfs, ignore_index=True)
    loss = get_loss(simulated, empirical, coherences, emphases=emphases)
    return loss


if __name__ == '__main__':

    label = sys.argv[1]
    study_name = f"hanks_{label}"
    optuna_trials = 10000

    # objective(None)
    # raise

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study = optuna.create_study(
        study_name=study_name,
        # storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial), n_trials=optuna_trials)