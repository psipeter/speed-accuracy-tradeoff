import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector
import json

if __name__ == '__main__':


    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"
    label = sys.argv[1
    best_params = {}
    emp = pd.read_pickle("data/fiedler_trial.pkl")

    for pid in emp['id'].unique():
        study_name=f"{pid}_{label}"
        study = optuna.create_study(
            study_name=study_name,
            storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
            load_if_exists=True,
            direction="minimize")
        print(f"pid {pid}, best trial {study.best_trial.number} / {len(study.trials)}, value {study.best_value}")
        best_params[pid] = study.best_params

    with open(f"data/fiedler_optimized_params_{label}.json", 'w') as f:
        json.dump(best_params, f, indent=1)
