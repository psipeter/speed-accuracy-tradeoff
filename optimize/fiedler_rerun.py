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
sid = pid
trained_difficulty = sys.argv[2]
label = sys.argv[3]
study_name = f"{pid}_{trained_difficulty}_{label}"
collect_name = f"{trained_difficulty}_{label}"
dPs = [0.4, 0.2, 0.1]
experiment_time = 6
dt = 0.001
perception_seed = 0
nNeurons = 500
rA = 1.5
dt_sample = 0.1
max_cues = 12
cue_step = 4

with open(f"data/fiedler_collect_{trained_difficulty}_{label}.json") as f:
    params = json.load(f)
param = params[pid]
ramp = param['ramp']
threshold = param['threshold']
relative = param['relative']
columns1 = ['type', 'pid', 'id', 'difficulty', 'trial', 'position', 'cue', 'value', 'fraction_sampled', 'sampled_cues', 'max_cues', 
        'chosen', 'target', 'accuracy', 'cue_choice_aligned']
columns2 = ['type', 'id', 'dP', 'trial', 'accuracy', 'cues', 'max_cues']
dfs1 = []
dfs2 = []

for dP in dPs:
    print(f"pid {pid}, dP {dP}")
    if dP==0.4: difficulty = 'easy'
    if dP==0.2: difficulty = 'moderate'
    if dP==0.1: difficulty = 'hard'
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
        accuracy = 1.0 if choice==net.inputs.correct else 0.0
        chosen = "L" if choice==0 else "R"
        target = "L" if net.inputs.correct==0 else "R"
        # print(choice, RT, dt_sample, max_cues, trial, total_time)
        cues = min(int(RT/dt_sample)+1, 2*max_cues)
        Ls = net.inputs.sampled[0]
        Rs = net.inputs.sampled[1]
        # print('L', Ls)
        # print("R", Rs)
        for p in range(cues):
            cue = 'L' if (p % 2 == 0) else 'R'  # L always shown for 500ms, then R is shown for 500ms in our model 
            if cue == 'R':
                value = Rs[int(p/2)]
            elif cue == 'L':
                value = Ls[int(p/2)]
            fraction_sampled = (p+1) / cues
            cue_choice_aligned = 0.0
            if chosen=='R':
                if (cue=='R' and value==1) or (cue=='L' and value==0):
                    cue_choice_aligned += 1
                else:
                    cue_choice_aligned += -1
            elif chosen=='L':
                if (cue=='L' and value==1) or (cue=='R' and value==0):
                    cue_choice_aligned += 1
                else:
                    cue_choice_aligned += -1
            # print(p, cue, value, fraction_sampled)
            df1 = pd.DataFrame([[
                'model', pid, sid, difficulty, trial, p, cue, value, fraction_sampled, cues, max_cues,
                chosen, target, accuracy, cue_choice_aligned
                ]], columns=columns1)
            dfs1.append(df1)
        if cues==0:  # participant chose before sampling any cues
            df1 = pd.DataFrame([[
                'model', pid, sid, difficulty, trial, -1, None, None, None, 0, max_cues, chosen, target, accuracy, None
                ]], columns=columns1)
            dfs1.append(df1)

        dfs2.append(pd.DataFrame([['model', sid, dP, trial, 100*accuracy, cues, max_cues]], columns=columns2))
        print(f"pid {pid}, trial {trial}, dP {dP}, elapsed time {total_time}, cues {cues}, choice {choice}, correct {net.inputs.correct}")
        total_time += RT
        trial += 1

sim1 = pd.concat(dfs1, ignore_index=True)
sim2 = pd.concat(dfs2, ignore_index=True)
sim1.to_pickle(f"data/fiedler1_rerun_{pid}_{trained_difficulty}_{label}.pkl")
sim2.to_pickle(f"data/fiedler2_rerun_{pid}_{trained_difficulty}_{label}.pkl")