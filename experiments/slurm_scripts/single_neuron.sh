#!/bin/bash
nvidia-smi
python -c "import torch;print(torch.cuda.is_available())"

cd ..
experiment_path="single_neuron"
save_path="../results/$experiment_path/"
echo $save_path

python single_neuron.py --save-path $save_path --run-id $1