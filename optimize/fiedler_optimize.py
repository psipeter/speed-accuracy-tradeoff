import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector
from model import SequentialPerception, build_network

def chi_squared_distance(a,b):
    distance = 0
    for i in range(len(a)):
        if a[i]+b[i]==0:
            continue
        else:
            distance += np.square(a[i] - b[i]) / (a[i]+b[i])
    return distance

def get_loss(simulated, empirical, dPs, max_cues, cue_step):
    loss = 0
    bins = np.arange(0.0, 2*max_cues+cue_step, cue_step)
    for dP in dPs:
        cues_sim = simulated.query("dP==@dP")['cues'].to_numpy()
        cues_emp = empirical.query("dP==@dP")['cues'].to_numpy()
        hist_cues_sim = np.histogram(cues_sim, bins=bins)[0]
        hist_cues_emp = np.histogram(cues_emp, bins=bins)[0]
        normed_hist_cues_sim = hist_cues_sim / len(cues_sim)
        normed_hist_cues_emp = hist_cues_emp / len(cues_emp)
        loss += chi_squared_distance(normed_hist_cues_sim, normed_hist_cues_emp)
    print(f"loss {loss}")
    return loss

def objective(trial, pid):
    perception_seed = 0
    dt = 0.001
    dt_sample = 0.1
    experiment_time = 60
    # Optuna picks optimized parameters
    ramp = trial.suggest_float("ramp", 0.5, 1.5, step=0.01)
    relative = trial.suggest_float("relative", 0.1, 1.0, step=0.01)
    threshold = trial.suggest_float("threshold", 0.1, 0.5, step=0.01)
    # We fix the remaining parameters, but save them for our records
    nNeurons = trial.suggest_categorical("nNeurons", [500])
    rA = trial.suggest_categorical("radius", [1.0])
    max_cues = trial.suggest_categorical("max_cues", [12])
    cue_step = trial.suggest_categorical("cue_step", [5])  # used in loss function for binning data
    dPs = trial.suggest_categorical("dPs", [[0.2]])  # change this to compute lost over different difficulties

    dfs = []
    for dP in dPs:
        print(f"dP {dP}, ramp {ramp}, threshold {threshold}, relative {relative}")
        # run task_trials iterations of the task, measuring simulated reaction times and accuracies
        inputs = SequentialPerception(dt_sample=dt_sample, seed=perception_seed+int(dP*10), max_cues=max_cues)
        total_time = 0
        task_trial = 0
        while total_time < experiment_time:
            inputs.create(dP=dP)
            # net = build_network(inputs, seed=int(pid), ramp=ramp, threshold=threshold, nNeurons=nNeurons, relative=relative)
            net = build_network(inputs, seed=task_trial, ramp=ramp, threshold=threshold, nNeurons=nNeurons, relative=relative, rA=rA)        
            sim = nengo.Simulator(net, progress_bar=False)
            choice = None
            RT = None
            while choice==None:
                sim.run(dt)
                if np.any(sim.data[net.pAction][-1,:] > 0.01):
                    choice = np.argmax(sim.data[net.pAction][-1,:])
                    RT = sim.trange()[-1]
                if sim.trange()[-1] > 2*max_cues*dt_sample-dt:
                    choice = np.argmax(sim.data[net.pValue][-1,:])
                    RT = sim.trange()[-1]
            acc = 100 if choice==inputs.correct else 0
            cues = min(int(RT/dt_sample)+1, 2*max_cues)
            print(f"trial {task_trial}, dP {dP}, elapsed time {total_time}, cues {cues}")
            dfs.append(pd.DataFrame([['model', pid, dP, task_trial, cues, acc]], columns=('type', 'pid', 'dP', 'trial', 'cues', 'accuracy')))
            total_time += RT
            task_trial += 1
    simulated = pd.concat(dfs, ignore_index=True)
    empirical = pd.read_pickle("data/fiedler_trial.pkl").query("id==@pid & max_cues==@max_cues")
    loss = get_loss(simulated, empirical, dPs, max_cues, cue_step)
    return loss


if __name__ == '__main__':

    pid = int(sys.argv[1])
    label = sys.argv[2]
    study_name = f"{pid}_{label}"
    optuna_trials = 200

    # objective(None, dPs, pid)
    # raise

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")
    study.optimize(lambda trial: objective(trial, pid), n_trials=optuna_trials)