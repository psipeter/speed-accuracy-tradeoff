import sys
import numpy as np
import pandas as pd
import pickle

trained_difficulty = sys.argv[1]
label = sys.argv[2]

dfs1 = []
dfs2 = []
for pid in range(2):
    study_name = f"{pid}_{trained_difficulty}_{label}"
    df1 = pd.read_pickle(f"data/fiedler_position_{study_name}.pkl")   
    df2 = pd.read_pickle(f"data/fiedler_trial_{study_name}.pkl")   
    dfs1.append(df1)
    dfs2.append(df2)

data1 = pd.concat(dfs1, ignore_index=True)
data2 = pd.concat(dfs2, ignore_index=True)
data1.to_pickle(f"data/fiedler_position_{trained_difficulty}_{label}.pkl")
data2.to_pickle(f"data/fiedler_trial_{trained_difficulty}_{label}.pkl")