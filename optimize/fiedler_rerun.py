import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector
import json
from network_revised import SequentialPerception, build_network

pid = str(sys.argv[1])
difficulty = sys.argv[2]
label = sys.argv[3]
study_name = f"{pid}_{difficulty}_{label}"
collect_name = f"{difficulty}_{label}"
dPs = [0.4, 0.2, 0.1]
experiment_time = 60
dt = 0.001
perception_seed = 0
nNeurons = 500
rA = 1.5
dt_sample = 0.1
max_cues = 12
cue_step = 4

with open(f"data/fiedler_collect_{difficulty}_{label}.json") as f:
    params = json.load(f)
param = params[pid]
ramp = param['ramp']
threshold = param['threshold']
relative = param['relative']
columns = ('type', 'id', 'dP', 'trial', 'accuracy', 'cues', 'max_cues')
dfs = []

for dP in dPs:
    print(f"pid {pid}, dP {dP}")
    inputs = SequentialPerception(seed=perception_seed+int(dP*10), dt_sample=dt_sample, max_cues=max_cues)
    total_time = 0
    trial = 0
    while total_time < experiment_time:
        inputs.create(dP=dP)
        net = build_network(inputs, seed=trial, ramp=ramp, threshold=threshold, nNeurons=nNeurons, relative=relative, rA=rA)
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
        correct = 1 if choice==net.inputs.correct else 0
        cues = min(int(RT/dt_sample)+1, 2*max_cues)
        dfs.append(pd.DataFrame([['model', pid, dP, trial, 100*correct, cues, max_cues]], columns=columns))
        print(f"pid {pid}, trial {trial}, dP {dP}, elapsed time {total_time}, cues {cues}, choice {choice}, correct {net.inputs.correct}")
        total_time += RT
        trial += 1
sim = pd.concat(dfs, ignore_index=True)
sim.to_pickle(f"data/fiedler_rerun_{pid}_{difficulty}_{label}.pkl")