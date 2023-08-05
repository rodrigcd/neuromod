nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="hr_maml_sweep_v2"
save_path="../results/$experiment_path/"
echo $save_path

python maml.py --save-path $save_path --run-id $1