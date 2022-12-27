nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="task_switch_main"
save_path="../results/$experiment_path/"
echo $save_path

python task_switch.py --run-name "slow_switch_run$1" --datasets AffineCorrelatedGaussian --save-path $save_path\
 --change-task-every 1600 --iter-control 500
python task_switch.py --run-name "slow_switch_run$1" --datasets MNIST_shared --save-path $save_path\
 --change-task-every 6000 --n-steps 36000 --iter-control 500
python task_switch.py --run-name "slow_switch_run$1" --datasets MNIST_diff --save-path $save_path\
 --change-task-every 6000 --n-steps 36000 --iter-control 500