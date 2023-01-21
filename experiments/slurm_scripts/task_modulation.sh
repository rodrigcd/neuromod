nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="task_modulation"
save_path="../results/$experiment_path/"
echo $save_path

python task_modulation.py --run-name "run_id_$1" --save-path $save_path --run-id $1