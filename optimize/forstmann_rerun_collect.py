import sys
import numpy as np
import pandas as pd
import pickle

label = sys.argv[1]
emp = pd.read_pickle(f"data/forstmann2011.pkl").query("age=='young'")
dfs = []
for pid in emp['pid'].unique():
    dfs.append(pd.read_pickle(f"data/forstmann_rerun_{pid}_{label}.pkl"))
simulated = pd.concat(dfs, ignore_index=True)
simulated.to_pickle(f"data/forstmann_{label}.pkl")