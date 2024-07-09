import sys
import numpy as np
import pandas as pd
import pickle

label = sys.argv[1]
emp = pd.read_pickle("data/fiedler_trial.pkl")
dfs = []
for pid in emp['id'].unique():
    dfs.append(pd.read_pickle(f"data/fiedler_rerun_{pid}_{label}.pkl"))
data = pd.concat(dfs, ignore_index=True)
data.to_pickle(f"data/fiedler_rerun_{label}.pkl")