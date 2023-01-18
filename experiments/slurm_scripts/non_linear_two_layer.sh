nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="non_linear_clstr"
save_path="../results/$experiment_path/"
echo $save_path

python non_linear_two_layer.py --run-name non_linear --dataset "AffineCorrelatedGaussian" --save-path $save_path --iter-control 700
python non_linear_two_layer.py --run-name non_linear --dataset "Semantic" --save-path $save_path --iter-control 700
python non_linear_two_layer.py --run-name non_linear --dataset "MNIST" --save-path $save_path --iter-control 700
