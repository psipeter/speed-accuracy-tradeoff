import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import json
from model import SequentialPerception, build_network

pid = str(sys.argv[1])
label = sys.argv[2]
experiment_time = 60
dt = 0.001
perception_seed = 0
dt_sample = 0.1
amin = 0.1
nNeurons = 500
rA = 1.0

with open(f"data/fiedler_optimized_params_{label}.json") as f:
    params = json.load(f)
param = params[pid]
ramp = param['ramp']
threshold = param['threshold']
relative = param['relative']

columns = ['type', 'id', 'dP', 'trial', 'accuracy', 'cues', 'max_cues']
dfs = []
for max_cues in [12, 18]:
    for dP in [0.4, 0.2, 0.1]:
        print(f"dP {dP}, ramp {ramp}, threshold {threshold}, relative {relative}")
        # run task_trials iterations of the task, measuring simulated reaction times and accuracies
        inputs = SequentialPerception(dt_sample=dt_sample, seed=perception_seed+int(dP*10), max_cues=max_cues)
        total_time = 0
        task_trial = 0
        while total_time < experiment_time:
            inputs.create(dP=dP)
            net = build_network(inputs, seed=task_trial, ramp=ramp, threshold=threshold, nNeurons=nNeurons, relative=relative, rA=rA)        
            sim = nengo.Simulator(net, progress_bar=False)
            choice = None
            RT = None
            while choice==None:
                sim.run(dt)
                if np.any(sim.data[net.pAction][-1,:] > amin):
                    choice = np.argmax(sim.data[net.pAction][-1,:])
                    RT = sim.trange()[-1]
                if sim.trange()[-1] > 2*max_cues*dt_sample-dt:
                    choice = np.argmax(sim.data[net.pValue][-1,:])
                    RT = sim.trange()[-1]
            acc = 100 if choice==inputs.correct else 0
            cues = min(int(RT/dt_sample)+1, 2*max_cues)
            print(f"trial {task_trial}, dP {dP}, elapsed time {total_time}, cues {cues}")
            dfs.append(pd.DataFrame([['model', int(pid), dP, task_trial, acc, cues, max_cues]], columns=columns))
            total_time += RT
            task_trial += 1

simulated = pd.concat(dfs, ignore_index=True)
simulated.to_pickle(f"data/fiedler_rerun_{pid}_{label}.pkl")