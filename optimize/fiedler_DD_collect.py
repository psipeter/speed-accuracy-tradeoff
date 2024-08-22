import sys
import numpy as np
import pandas as pd
import pickle
import json

max_cues = 12
emp = pd.read_pickle(f"data/fiedler_trial.pkl").query("max_cues==@max_cues")
data = []
params = {}
for pid in emp['pid'].unique():
    data.append(pd.read_pickle(f"data/fiedler_rerun_DD_{pid}.pkl"))
    with open(f"data/fiedler_params_DD_{pid}.json") as f:
        p = json.load(f)
    params[pid] = p
reruns = pd.concat(data, ignore_index=True)
reruns.to_pickle(f"data/fiedler_reruns_DD.pkl")
with open(f"data/fiedler_params_DD.json", 'w') as f:
    json.dump(params, f, indent=1)