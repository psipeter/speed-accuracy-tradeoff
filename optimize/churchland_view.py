import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector

if __name__ == '__main__':

    # nActions = int(sys.argv[1])
    label = sys.argv[1]
    study_name = f"churchland_{label}"

    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
        load_if_exists=True,
        direction="minimize")

    print("Trials Simulated: ", len(study.trials))
    print("Best Trial: ", study.best_trial.number)
    print("Best Parameters: ", study.best_params)
    print("Best Value: ", study.best_value)