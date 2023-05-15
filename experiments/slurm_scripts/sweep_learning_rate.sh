nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="sweep_learning_rate_v2"
save_path="../results/$experiment_path/"
echo $save_path

python sweep_learning_rate.py --run-name "run_id_$1" --save-path $save_path --run-id $1