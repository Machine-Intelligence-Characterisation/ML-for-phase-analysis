#!/bin/bash
#SBATCH --job-name=ML_Training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00

#SBATCH --output=/monfs01/projects/ys68/XRD_phase_analysis/using_Monash_HPC/run_monARCH_GPU/slurm_outputs/SLURM%j.out

# Load necessary modules
module load cuda

nvidia-smi
deviceQuery

srun python /monfs01/projects/ys68/XRD_phase_analysis/scripts/training/main_training.py