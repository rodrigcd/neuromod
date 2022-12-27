nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="long_run_non_linear"
save_path="../results/$experiment_path/"
echo $save_path

python linear_two_layer.py --dataset "AffineCorrelatedGaussian" --n-steps 200 --iter-control 10
python linear_two_layer.py --dataset "Semantic" --n-steps 200 --iter-control 10
python linear_two_layer.py --dataset "MNIST" --n-steps 200 --iter-control 10