import sys
import pandas as pd

trained_difficulty = sys.argv[1]
label = sys.argv[2]
emp = pd.read_pickle("data/fiedler_trials.pkl")

for pid in emp['id'].unique():
   string = f"python fiedler_rerun.py {pid} {trained_difficulty} {label}"
   with open (f'{pid}_rerun.sh', 'w') as rsh:
       rsh.write('''#!/bin/bash''')
       rsh.write("\n")
       rsh.write('''#SBATCH --mem=8G''')
       rsh.write("\n")
       rsh.write('''#SBATCH --nodes=1''')
       rsh.write("\n")
       rsh.write('''#SBATCH --ntasks-per-node=1''')
       rsh.write("\n")
       rsh.write('''#SBATCH --time=2:0:0''')
       rsh.write("\n")
       rsh.write('''module load python/3.11.2 scipy-stack mysql''')
       rsh.write("\n")
       rsh.write('''source ~/ENV311/bin/activate''')
       rsh.write("\n")
       rsh.write('''cd ~/projects/def-celiasmi/psipeter/speed-accuracy-tradeoff''')
       rsh.write("\n")
       rsh.write(string)