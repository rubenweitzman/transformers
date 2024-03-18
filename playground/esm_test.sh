#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1 #commented out, can use any node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --output=/users/rubman/Desktop/transformers/playground/slurm/%x-%j.out
#SBATCH --error=/users/rubman/Desktop/transformers/playground/slurm/%x-%j.err
#SBATCH --job-name="sdpa_esm"
#SBATCH --mem=480GB
#SBATCH --exclude=clpc144,clpc155,clpc156,clpc158


export CONDA_ENVS_PATH=/scratch-ssd/pastin/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/pastin/conda_pkgs
# source /scratch-ssd/oatml/miniconda3/bin/activate protein_npt_env
source /scratch-ssd/oatml/miniconda3/bin/activate /users/rubman/.conda/envs/protriever


srun python playground/debug_esm.py 
# srun python playground/debug_bart.py 