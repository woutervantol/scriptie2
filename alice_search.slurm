#!/bin/bash
#SBATCH --job-name=RayTune
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="tol@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu-long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G


### load relevant modules
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
source /home/s2041340/Thesis/python_venv_alice/bin/activate


echo "[$SHELL] #### Starting Python test on $(date)"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] ## current working directory: "$CWD



### run single model for all variations
for MODEL in "HYDRO_FIDUCIAL" "HYDRO_JETS_published" "HYDRO_STRONG_AGN" "HYDRO_STRONG_JETS_published" "HYDRO_STRONG_SUPERNOVA" "HYDRO_STRONGER_AGN" "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA" "HYDRO_STRONGEST_AGN" "HYDRO_WEAK_AGN"
do
    echo $MODEL
    python /home/s2041340/Thesis/scriptie2/tune_search.py -m $MODEL -b 3.5
    python /home/s2041340/Thesis/scriptie2/tune_search.py -m $MODEL -b 3.5 -n True
done

### run combinations
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_STRONG_AGN -b 12 -t all_but
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 12 -t all_but
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 12 -t all
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 12 -t extremes
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_STRONG_AGN -b 12 -t all_but -n True
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 12 -t all -n True
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 12 -t extremes -n True
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 12 -t all_but -n True

### run single channel fiducial
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 3.5 -c low
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 3.5 -c high
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 3.5 -c low -n True
python /home/s2041340/Thesis/scriptie2/tune_search.py -m HYDRO_FIDUCIAL -b 3.5 -c high -n True


echo "[$SHELL] ## Script finished"