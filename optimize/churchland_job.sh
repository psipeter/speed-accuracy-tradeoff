#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:0:0

module load python/3.11.2 scipy-stack mysql
source ~/ENV311/bin/activate
cd ~/projects/def-celiasmi/psipeter/speed-accuracy-tradeoff
python churchland_optimize.py dec14rel