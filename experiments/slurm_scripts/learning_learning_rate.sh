nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="corrected_learning_rate"
save_path="../results/$experiment_path/"
echo $save_path

python learning_rate.py --run-name "run_id_$1" --save-path $save_path --run-id $1
#python learning_rate.py --run-name "run_id_3" --save-path $save_path --run-id 3
#python learning_rate.py --run-name "run_id_8" --save-path $save_path --run-id 8
#python learning_rate.py --run-name "run_id_13" --save-path $save_path --run-id 13