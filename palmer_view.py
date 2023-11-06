import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector

if __name__ == '__main__':

    name = sys.argv[1]
    condition = sys.argv[2]
    label = sys.argv[3]

    # study = optuna.create_study(
    #       study_name=f"{name}_{condition}_{label}",
    #       load_if_exists=True,
    #       direction="minimize")

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = ""

    study = optuna.create_study(
        study_name=f"{name}_{condition}_{label}",
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial.number)