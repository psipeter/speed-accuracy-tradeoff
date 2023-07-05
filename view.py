import sys
import numpy as np
import nengo
import pandas as pd
import pickle
import optuna
import mysql.connector

if __name__ == '__main__':

    pid = int(sys.argv[1])
    study_name = sys.argv[2]
    host = "gra-dbaas1.computecanada.ca"
    user = "psipeter"
    password = "gimuZhwLKPeU99bt"

    study = optuna.create_study(
          study_name=f"{study_name}",
          storage=f"mysql+mysqlconnector://{user}:{password}@{host}/{user}_{study_name}",
          load_if_exists=True,
          direction="minimize")

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial.number)