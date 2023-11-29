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
    difficulty = sys.argv[1]
    label = sys.argv[2]
    best_params = {}

    for pid in range(57):
        study_name=f"{pid}_{difficulty}_{label}"
        study = optuna.create_study(
            study_name=study_name,
            storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
            load_if_exists=True,
            direction="minimize")
        print(f"pid {pid}, best trial {study.best_trial.number} / {len(study.trials)}, value {study.best_value}")
        best_params[pid] = study.best_params

    with open(f"data/fiedler_collect_{label}.json", 'w') as f:
        json.dump(best_params, f, indent=1)
