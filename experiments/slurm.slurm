#!/bin/bash
#
#SBATCH --job-name=test_switch
#SBATCH -w gpu-380-18
#SBATCH --output=test_arr.out
#SBATCH --error=test_arr.err
#SBATCH -p gpu 
#SBATCH -N 1
#SBATCH --mem=4G
#SBATCH --time=0-16:00
#SBATCH --gres=gpu:1  # this is important to use GPUs
#

# source /usr/share/modules/init/bash
source /usr/share/modules/init/bash
module load cuda/11.6
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.6.2_510.47.03

eval "$(conda shell.bash hook)"
conda activate metamod

# source activate metamod
nvidia-smi

python -c "import torch;print(torch.cuda.is_available())"

python slow_task_switch.py --run-name slow_switch_ --datasets AffineCorrelatedGaussian --save-path "../results/cluster_RTX/" --change-task-every 1500
python slow_task_switch.py --run-name slow_switch_ --datasets MultiDimGaussian --save-path "../results/cluster_RTX/" --change-task-every 1500
python slow_task_switch.py --run-name mid_switch_ --datasets AffineCorrelatedGaussian --save-path "../results/cluster_RTX/" --change-task-every 600
python slow_task_switch.py --run-name mid_switch_ --datasets MultiDimGaussian --save-path "../results/cluster_RTX/" --change-task-every 600
# python3.7 slow_task_switch.py --run-name fast_switch_ --datasets AffineCorrelatedGaussian --save-path ../results/task_switch2/ --change-task-every 150
# python3.7 slow_task_switch.py --run-name fast_switch_ --datasets MultiDimGaussian --save-path ../results/task_switch2/ --change-task-every 150

