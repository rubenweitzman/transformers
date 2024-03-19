#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1 #commented out, can use any node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH -p gpu_quad,gpu,gpu_marks,gpu_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=gpuquad_qos
#SBATCH --time=48:00:00
#SBATCH --output=/home/ruw846/transformers/playground/slurm/%x-%j.out
#SBATCH --error=/home/ruw846/transformers/playground/slurm/%x-%j.err
#SBATCH --job-name="sdpa_bart"
#SBATCH --mem=40GB


set -e #o pipefail # fail fully on first line failure (from Joost slurm_for_ml)


# module load miniconda3/4.10.3
module load gcc/9.2.0
module load cuda/12.1  # Or 11.7. But PoET wants 11.8 actually..
# export NCCL_DEBUG=INFO

source /home/ruw846/miniforge3/etc/profile.d/conda.sh
conda activate protriever



srun python playground/debug_bart.py 
# srun python playground/debug_bart.py 