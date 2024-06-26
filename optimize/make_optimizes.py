import sys
import subprocess
import pandas as pd

label = sys.argv[1]
emp = pd.read_pickle("data/fiedler_trial.pkl")

for pid in emp['id'].unique():
   delete_string = f"drop database psipeter_{pid}_{label}"
   create_string = f"create database psipeter_{pid}_{label}"
   # a = subprocess.run(["mysql", "-u", "psipeter", "-e", delete_string])
   b = subprocess.run(["mysql", "-u", "psipeter", "-e", create_string])
   string = f"python fiedler_optimize.py {pid} {label}"
   with open (f'{pid}_optimize.sh', 'w') as rsh:
       rsh.write('''#!/bin/bash''')
       rsh.write("\n")
       rsh.write('''#SBATCH --mem=8G''')
       rsh.write("\n")
       rsh.write('''#SBATCH --nodes=1''')
       rsh.write("\n")
       rsh.write('''#SBATCH --ntasks-per-node=1''')
       rsh.write("\n")
       rsh.write('''#SBATCH --time=8:0:0''')
       rsh.write("\n")
       rsh.write('''module load python/3.11.2 scipy-stack mysql''')
       rsh.write("\n")
       rsh.write('''source ~/ENV311/bin/activate''')
       rsh.write("\n")
       rsh.write('''cd ~/projects/def-celiasmi/psipeter/speed-accuracy-tradeoff''')
       rsh.write("\n")
       rsh.write(string)