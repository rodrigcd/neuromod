#!/bin/bash
#
#SBATCH --job-name=test_switch
#SBATCH --output=test_arr.out
#SBATCH --error=test_arr.err
#SBATCH -p gpu 
#SBATCH -N 1
#SBATCH --mem=4G
#SBATCH --time=0-16:00
##SBATCH --gres=gpu:1  # this is important to use GPUs
#

# conda activate atari
python3.7 slow_task_switch.py --run-name slow_switch_ --datasets AffineCorrelatedGaussian --save-path "../results/task_switch2/" --change-task-every 1500
python3.7 slow_task_switch.py --run-name slow_switch_ --datasets MultiDimGaussian --save-path "../results/task_switch2/" --change-task-every 1500
python3.7 slow_task_switch.py --run-name mid_switch_ --datasets AffineCorrelatedGaussian --save-path "../results/task_switch2/" --change-task-every 600
python3.7 slow_task_switch.py --run-name mid_switch_ --datasets MultiDimGaussian --save-path "../results/task_switch2/" --change-task-every 600
# python3.7 slow_task_switch.py --run-name fast_switch_ --datasets AffineCorrelatedGaussian --save-path ../results/task_switch2/ --change-task-every 150
# python3.7 slow_task_switch.py --run-name fast_switch_ --datasets MultiDimGaussian --save-path ../results/task_switch2/ --change-task-every 150

