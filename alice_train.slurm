#!/bin/bash

#SBATCH --job-name=TestBest
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="tol@strw.leidenuniv.nl"
#SBATCH --mail-type="NONE"
#SBATCH --time=01:00:00
#SBATCH --partition=testing 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=600M

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
source /home/s2041340/Thesis/python_venv_alice/bin/activate

echo "[$SHELL] #### Starting Python test on $(date)"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] ## current working directory: "$CWD


echo "[$SHELL] ## Run script"
python /home/s2041340/Thesis/scriptie2/train_network.py
echo "[$SHELL] ## Script finished"