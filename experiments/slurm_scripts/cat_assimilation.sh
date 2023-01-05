nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="category_assimilation"
save_path="../results/$experiment_path/"
echo $save_path

python category_assimilation.py --run-name "run_id_$1" --save-path $save_path --run-id $1