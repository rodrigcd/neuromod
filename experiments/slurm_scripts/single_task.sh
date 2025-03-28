nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="single_task_more_iters"
save_path="../results/$experiment_path/"
echo $save_path

python linear_two_layer.py --dataset "AffineCorrelatedGaussian" --run-name "run_id_$1" --save-path $save_path\
 --n-steps 16000 --iter-control 1000
python linear_two_layer.py --dataset "Semantic" --run-name "run_id_$1" --save-path $save_path\
 --n-steps 16000 --iter-control 1000
python linear_two_layer.py --dataset "MNIST" --run-name "run_id_$1" --save-path $save_path\
 --n-steps 16000 --iter-control 1000