#!/bin/bash

#SBATCH --job-name=Jupyter_notebook
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="tol@strw.leidenuniv.nl"
#SBATCH --mail-type="NONE"
#SBATCH --time=01:00:00
#SBATCH --partition=testing 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

module load IPython/7.9.0-foss-2019b-Python-3.7.4

echo "[$SHELL] #### Starting Python test on $(date)"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] ## current working directory: "$CWD

IPADDR = $(hostname -i)

echo "[$SHELL] ## Starting notebook"
srun jupyter notebook --ip=$IPADDR --no-browser --port=8989
echo "[$SHELL] ## Script finished"