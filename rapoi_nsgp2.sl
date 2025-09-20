#!/bin/bash

#SBATCH --time=0-24:00:00         # Walltime (HH:MM:SS)  7d+4hr
#SBATCH --mem=40G                     # Memory in MB
#SBATCH --cpus-per-task=40
#SBATCH --partition=parallel         # Will request 70 logical CPUs per task.
#SBATCH --output=outputs/%x_%j.out

module load GCCcore/13.2.0 Python/3.11.5
source ../../zhengxin/venv/bin/activate

python3 -m z.nsgp.main -r $1 -s $1 # --log-fronts-gen 69 --training-case 2
