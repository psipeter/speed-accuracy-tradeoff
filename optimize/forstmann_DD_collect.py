import sys
import numpy as np
import pandas as pd
import pickle
import json

emp = pd.read_pickle(f"data/forstmann2011.pkl").query("age=='young'")
data = []
params = {}
# for pid in emp['pid'].unique():
for pid in ['as1t', 'bd6t']:
    data.append(pd.read_pickle(f"data/forstmann_rerun_DD_{pid}.pkl"))
    with open(f"data/forstmann_params_DD_{pid}.json") as f:
        p = json.load(f)
    params[pid] = p
reruns = pd.concat(data, ignore_index=True)
reruns.to_pickle(f"data/forstmann_reruns_DD.pkl")
with open(f"data/forstmann_params_DD.json", 'w') as f:
    json.dump(params, f, indent=1)