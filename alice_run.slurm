#!/bin/bash

#SBATCH --job-name=test_python_simple
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="tol@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=100M
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module load Python/3.10.8-GCCcore-12.2.0
source $HOME/Thesis/scriptie2/python_venv_alice/bin/activate

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] ## current working directory: "$CWD


echo "[$SHELL] ## Run script"
python $HOME/Thesis/scriptie2/create_data.py
echo "[$SHELL] ## Script finished"l