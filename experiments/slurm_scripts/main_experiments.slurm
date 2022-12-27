#!/bin/bash
#
#SBATCH --job-name=main_exp
#SBATCH --output=logs/out_main_exp.log
#SBATCH --error=logs/err_main_exp.log
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=4G
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:1  # this is important to use GPUs
#SBATCH --array=0-4

source /usr/share/modules/init/bash
module load cuda/11.6
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.6.2_510.47.03
eval "$(conda shell.bash hook)"
conda activate metamod

# Linear network experiment
bash single_task.sh $SLURM_ARRAY_TASK_ID