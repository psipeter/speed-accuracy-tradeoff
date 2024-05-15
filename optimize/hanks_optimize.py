import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector
from model import DotPerception, build_network

def chi_squared_distance(a,b):
    distance = 0
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    return distance

def get_loss(simulated, empirical, coherences, emphases):
    total_loss = 0
    bins = np.arange(0.0, 1.5, 0.1)
    for emphasis in emphases:
        for coherence in coherences:
            coh = 100*coherence
            rts_sim = simulated.query("emphasis==@emphasis & coherence==@coherence")['RT'].to_numpy()
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
            total_loss += mean_loss
    return total_loss

def objective(trial):
    # let optuna choose the next parameters
    emphases = ['accuracy']
    name = 'E'
    task_trials = 10
    dt = 0.001
    tmax = 2
    coherences = [0.032, 0.064, 0.128, 0.256, 0.512]
    perception_seed = 0
    # network_seed = 0
    ramp = trial.suggest_float("ramp", 0.5, 1.5, step=0.01)
    threshold = trial.suggest_float("threshold", 0.01, 1.0, step=0.01)
    relative = trial.suggest_float("relative", 0.01, 1.0, step=0.01)
    dt_sample = trial.suggest_float("dt_sample", 0.001, 0.1, step=0.001)
    sigma = trial.suggest_float("sigma", 0.01, 0.8, step=0.01)
    nNeurons = trial.suggest_categorical("nNeurons", [500])
    rA = trial.suggest_categorical("radius", [1.0])
    minRate = trial.suggest_categorical("minRate", [60])
    maxRate = trial.suggest_categorical("maxRate", [80])
    max_rates = nengo.dists.Uniform(minRate, maxRate)
    # print(f"ramp {ramp}, threshold {threshold}, relative {relative}, dt_sample {dt_sample}, sigma {sigma}")
    empirical = pd.read_pickle("data/hanks2014_behavior.pkl").query("id==@name")

    # run task_trials iterations of the task, measuring simulated reaction times and accuracies
    dfs = []
    columns = ['emphasis', 'coherence', 'trial', 'RT', 'accuracy']
    for e, emphasis in enumerate(emphases):
        inputs = DotPerception(nActions=2, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
        for coherence in coherences:
            # print(f"coherence {coherence}")
            for t in range(task_trials):
                inputs.create(coherence=coherence)
                net = build_network(inputs, nActions=2, seed=t, ramp=ramp, threshold=threshold, relative=relative,
                                    nNeurons=nNeurons, max_rates=max_rates, rA=rA)
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
                dfs.append(pd.DataFrame([[emphasis, coherence, t, RT, acc]], columns=columns))
                print(f"emphasis {emphasis}, coherence {coherence}, trial {t}, RT {RT}")
    simulated = pd.concat(dfs, ignore_index=True)
    loss = get_loss(simulated, empirical, coherences, emphases)
    return loss


if __name__ == '__main__':

    label = sys.argv[1]
    study_name = f"hanks_{label}"
    optuna_trials = 2000

    # objective(None, "E")
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