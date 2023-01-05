#!/bin/bash
#
#SBATCH --job-name=cat_main
#SBATCH --output=logs/category_assimilation.log
#SBATCH --error=logs/category_assimilation.log
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=6G
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:1  # this is important to use GPUs
#SBATCH --array=0-9

source /usr/share/modules/init/bash
module load cuda/11.6
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.6.2_510.47.03
eval "$(conda shell.bash hook)"
conda activate metamod

# Linear network experiment
# bash single_task.sh $SLURM_ARRAY_TASK_ID

# Task switch
#bash task_switch.sh $SLURM_ARRAY_TASK_ID

# Category Assimilation
bash cat_assimilation.sh $SLURM_ARRAY_TASK_ID