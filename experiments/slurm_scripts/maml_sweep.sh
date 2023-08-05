nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="hr_maml_last_step_masking"
save_path="../results/$experiment_path/"
echo $save_path

# python maml.py --save-path $save_path --run-id $1
python maml.py --save-path $save_path --run-id $1 --last-step True