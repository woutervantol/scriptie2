#!/bin/bash
#SBATCH --job-name=RayTune
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="tol@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=1-04:00:00
#SBATCH --partition=gpu-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G

# module load ALICE/default
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
source /home/s2041340/Thesis/python_venv_alice/bin/activate


echo "[$SHELL] #### Starting Python test on $(date)"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] ## current working directory: "$CWD


# python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL

# for MODEL in "HYDRO_FIDUCIAL" "HYDRO_JETS_published" "HYDRO_STRONG_AGN" "HYDRO_STRONG_JETS_published" "HYDRO_STRONG_SUPERNOVA" "HYDRO_STRONGER_AGN" "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA" "HYDRO_STRONGEST_AGN" "HYDRO_WEAK_AGN"
# # for CHANNEL in "low" "high"
# do
#     echo "[$SHELL] ## Run script $MODEL"
#     python /home/s2041340/Thesis/scriptie2/tune_search.py -m $MODEL -b 3.5
# done

python /home/s2041340/Thesis/scriptie2/tune_search.py -t all -b 12
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -t all_but -b 12

echo "[$SHELL] ## Script finished"