#!/bin/bash
#
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -J collect
#SBATCH -p shared
#SBATCH -t 0-4:00 # Runtime in D-HH:MM
#SBATCH --mem=10000 # Memory pool for all cores (see also --mem-per-cpu)

source new-modules.sh
module load Anaconda3/5.0.1-fasrc01
source activate ray_planes
python collect.py
source deactivate