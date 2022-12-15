import numpy as np
import matplotlib.pyplot as plt

from metamod.networks import FullGradNet
from metamod.tasks import SemanticTask
from metamod.control import BatchNetworkControl

run_name = "composition_of_tasks"
results_path = "../results"
results_dict = {}

n_steps = 2000
iter_control = 20
inner_loop_iters = n_steps
outer_loop_iters = iter_control

dataset_params = {"batch_size": 256,
                  "h_levels": 3}

dataset = SemanticTask(**dataset_params)

model_params = {"learning_rate": 5e-3,
                "hidden_dim": 8,
                "intrinsic_noise": 0.0,
                "reg_coef": 0.0,
                "input_dim": dataset.input_dim,
                "output_dim": dataset.output_dim,
                "W1_0": None,
                "W2_0": None,
                "keep_grads": True}

model = FullGradNet(**model_params)

x, y = dataset.sample_batch()
current_loss = model.train_step(x, y, g1_tilda=None, g2_tilda=None)
model.reset_weights(model.W1.detach().cpu().numpy(), model.W2.detach().cpu().numpy())

control_params = {"inner_loop_iters": inner_loop_iters,
                  "outer_loop_iters": outer_loop_iters,
                  "control_lower_bound": -1.0,
                  "control_upper_bound": 1.0,
                  "gamma": 0.99,
                  "cost_coef": 0.3,
                  "reward_convertion": 1.0,
                  "init_g": None,
                  "control_lr": 0.01}

net_controller = BatchNetworkControl(network=model, task=dataset,
                                     **control_params)

net_controller.outer_loop()
net_controller.outer_loop()
net_controller.outer_loop()