nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="long_run_non_linear"
save_path="../results/$experiment_path/"
echo $save_path

python non_linear_two_layer.py --run-name non_linear --datasets "AffineCorrelatedGaussian" --save-path $save_path --iter-control 500
python non_linear_two_layer.py --run-name non_linear --datasets "Semantic" --save-path $save_path --iter-control 500
python non_linear_two_layer.py --run-name non_linear --datasets "MNIST" --save-path $save_path --iter-control 500
