nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="cat_prop"
save_path="../results/$experiment_path/"
echo $save_path

python cat_prop.py --run-name "run_id_$1" --save-path $save_path --run-id $1 --beta 5.0
python cat_prop.py --run-name "run_id_$1" --save-path $save_path --run-id $1 --beta 0.3