import sys
import numpy as np
import pandas as pd
import pickle

difficulty = sys.argv[1]
label = sys.argv[2]

dfs = []
for pid in range(57):
    study_name = f"{pid}_{difficulty}_{label}"
    df = pd.read_pickle(f"data/fiedler_rerun_{study_name}.pkl")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.to_pickle(f"data/fiedler_rerun_all_{difficulty}_{label}.pkl")
