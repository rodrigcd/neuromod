#!/bin/bash
#
#SBATCH --job-name=non_linear
#SBATCH -w gpu-380-17
#SBATCH --output=log_non_linear.out
#SBATCH --error=log_non_linear.err
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
experiment_path="long_run_non_linear"
save_path="../results/$experiment_path/"
echo $save_path

python non_linear_two_layer.py --run-name non_linear --datasets AffineCorrelatedGaussian --save-path $save_path --iter-control 500
python non_linear_two_layer.py --run-name non_linear --datasets MultiDimGaussian --save-path $save_path --iter-control 500
