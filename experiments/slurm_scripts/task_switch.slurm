#!/bin/bash
#
#SBATCH --job-name=task_switch
#SBATCH -w gpu-380-18
#SBATCH --output=log_task_switch.out
#SBATCH --error=log_task_switch.err
#SBATCH -p gpu 
#SBATCH -N 1
#SBATCH --mem=4G
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:1  # this is important to use GPUs

source /usr/share/modules/init/bash
module load cuda/11.6
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.6.2_510.47.03
eval "$(conda shell.bash hook)"
conda activate metamod
nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="long_run_task_switch"
save_path="../results/$experiment_path/"
echo $save_path

python task_switch.py --run-name slow_switch --datasets AffineCorrelatedGaussian --save-path $save_path --change-task-every 1600 --iter-control 500
python task_switch.py --run-name slow_switch --datasets MultiDimGaussian --save-path $save_path --change-task-every 1600 --iter-control 500
python task_switch.py --run-name mid_switch --datasets AffineCorrelatedGaussian --save-path $save_path --change-task-every 800 --iter-control 500
python task_switch.py --run-name mid_switch --datasets MultiDimGaussian --save-path $save_path --change-task-every 800 --iter-control 500
python task_switch.py --run-name fast_switch --datasets AffineCorrelatedGaussian --save-path $save_path --change-task-every 150 --iter-control 500
python task_switch.py --run-name fast_switch --datasets MultiDimGaussian --save-path $save_path --change-task-every 150 --iter-control 500

