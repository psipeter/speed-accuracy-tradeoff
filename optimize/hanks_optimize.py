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

def objective(trial, name):
    # let optuna choose the next parameters
    emphases = ['accuracy']
    shared = ['ramp', 'threshold', 'relative']
    dt = 0.001
    task_trials = 400
    nNeurons = 500
    max_rates = nengo.dists.Uniform(60, 80)
    rA = 1.5
    dt_sample = 0.03
    sigma = 0.6
    tmax = 1.5
    coherences = [0.032, 0.064, 0.128, 0.256, 0.512]
    perception_seed = 0
    nActions = 2
    ramp1 = trial.suggest_float("ramp1", 0.5, 2.0, step=0.01)
    threshold1 = trial.suggest_float("threshold1", 0.01, 1.0, step=0.01)
    # relative1 = trial.suggest_float("relative1", 0.0, 1.0, step=0.01)
    relative1 = trial.suggest_float("relative1", 1.0, 1.0, step=0.01)
    print(f"ramp1 {ramp1}, threshold1 {threshold1}, relative1 {relative1}")
    ramp2 = ramp1 if 'ramp' in shared else trial.suggest_float("ramp2", 0.5, 2.0, step=0.01)
    threshold2 = threshold1 if 'threshold' in shared else trial.suggest_float("threshold2", 0.01, 1.0, step=0.01)
    relative2 = relative1 if 'relative' in shared else trial.suggest_float("relative2", 0.0, 1.0, step=0.01)     
    print(f"ramp2 {ramp2}, threshold2 {threshold2}, relative2 {relative2}")
    empirical = pd.read_pickle("data/hanks2014_behavior.pkl").query("id==@name")

    # run task_trials iterations of the task, measuring simulated reaction times and accuracies
    dfs = []
    for e, emphasis in enumerate(emphases):
        ramp = [ramp1, ramp2][e]
        threshold = [threshold1, threshold2][e]
        relative = [relative1, relative2][e]
        inputs = DotPerception(nActions=nActions, dt_sample=dt_sample, seed=perception_seed, sigma=sigma)
        for coherence in coherences:
            # print(f"coherence {coherence}")
            for t in range(task_trials):
                inputs.create(coherence=coherence)
                net = build_network(inputs, nActions=nActions, seed=t, ramp=ramp, threshold=threshold, relative=relative,
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
                dfs.append(pd.DataFrame([[emphasis, coherence, t, RT, acc]], columns=('emphasis', 'coherence', 'trial', 'RT', 'accuracy')))
                print(f"emphasis {emphasis}, coherence {coherence}, trial {t}, RT {RT}")
    simulated = pd.concat(dfs, ignore_index=True)
    loss = get_loss(simulated, empirical, coherences, emphases)
    return loss


if __name__ == '__main__':

    # nActions = int(sys.argv[1])
    name = sys.argv[1]
    label = sys.argv[2]
    study_name = f"hanks_{name}_{label}"
    optuna_trials = 2000

    # objective(None, "E")
    # raise

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, name), n_trials=optuna_trials)