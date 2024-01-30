import subprocess
pids = range(57)
for pid in pids:
   a = subprocess.run(["sbatch", f"{pid}_rerun.sh"])