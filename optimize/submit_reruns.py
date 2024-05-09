import subprocess
import pandas as pd
emp = pd.read_pickle("data/fiedler_trial.pkl")

for pid in emp['id'].unique():
   a = subprocess.run(["sbatch", f"{pid}_rerun.sh"])