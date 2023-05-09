#!/bin/bash
#
#SBATCH --job-name=lr_g
#SBATCH --output=logs/out_lr_g.log
#SBATCH --error=logs/err_lr_g.log
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=6G
#SBATCH --time=0-18:00
#SBATCH --gres=gpu:1  # this is important to use GPUs
#SBATCH --array=0-14  #0-140 for single neuron

source /usr/share/modules/init/bash
module load cuda/11.6
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.6.2_510.47.03
eval "$(conda shell.bash hook)"
conda activate metamod

# Single neuron experiment (Ask for 2 hours) 0-140 for single neuron
# bash single_neuron.sh $SLURM_ARRAY_TASK_ID

# Linear network experiment array 0-4
# bash single_task.sh $SLURM_ARRAY_TASK_ID

# Task switch array 0-4
# bash task_switch.sh $SLURM_ARRAY_TASK_ID

# Category Assimilation # 0-9 array
#bash cat_assimilation.sh $SLURM_ARRAY_TASK_ID

# Task engagement # 0-9 array
#bash task_engagement.sh $SLURM_ARRAY_TASK_ID

# Non-linear network  # Use Array 4
# bash non_linear_two_layer.sh

# CATEGORY ASSIMILATION CLASS PROPORTIONAL # SELECT BETA, THEN ARRAY 0-9
# bash cat_prop.sh $SLURM_ARRAY_TASK_ID

# Task Modulation 0-4 array
# bash task_modulation.sh $SLURM_ARRAY_TASK_ID

# Learning learning rate 0-14 array
bash learning_learning_rate.sh $SLURM_ARRAY_TASK_ID

# Maml easy experiment 0-9 array
# bash maml_sweep.sh $SLURM_ARRAY_TASK_ID