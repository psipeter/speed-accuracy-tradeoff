import sys
import numpy as np
import pandas as pd
import pickle

label = sys.argv[1]
emp = pd.read_pickle("data/fiedler_trial.pkl")

dfs1 = []
dfs2 = []
for pid in emp['id'].unique():
    df1 = pd.read_pickle(f"data/fiedler_position_{pid}_{label}.pkl")   
    df2 = pd.read_pickle(f"data/fiedler_trial_{pid}_{label}.pkl")   
    dfs1.append(df1)
    dfs2.append(df2)

data1 = pd.concat(dfs1, ignore_index=True)
data2 = pd.concat(dfs2, ignore_index=True)
data1.to_pickle(f"data/fiedler_position_{label}.pkl")
data2.to_pickle(f"data/fiedler_trial_{label}.pkl")

bins = np.arange(0.0, 1.2, 0.2)
columns = ['type', 'id', 'dP', 'max_cues', 'fraction_sampled', 'mean_cue_choice_aligned']
dfs = []
for pid in data2['id'].unique():
    for dP in data2['dP'].unique():
        for max_cues in data2['max_cues'].unique():
            sim = data2.query('id==@pid & dP==@dP & max_cues==@max_cues')
            for i in range(len(bins)-1):
                left = bins[i]
                right = bins[i+1]
                midpoint = (left + right) / 2
                cue_choices_aligned = sim.query('fraction_sampled>@left & fraction_sampled<=@right')['cue_choice_aligned'].to_numpy()
                mean = np.mean(cue_choices_aligned) if len(cue_choices_aligned)>1 else None
                df = pd.DataFrame([[
                    'model', pid, dP, max_cues, midpoint, mean,
                    ]], columns=columns)
                dfs.append(df)
sim = pd.concat(dfs, ignore_index=True)
sim.to_pickle(f"data/fiedler_binned_{label}.pkl")  # each row contains average and std data from one participant in one condition