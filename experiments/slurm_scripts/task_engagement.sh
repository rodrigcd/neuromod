nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="task_engagement"
save_path="../results/$experiment_path/"
echo $save_path

python task_engagement.py --run-name "run_id_$1" --save-path $save_path --run-id $1 --engage-type "active"
#python task_engagement.py --run-name "run_id_$1" --save-path $save_path --run-id $1 --engage-type "attention"