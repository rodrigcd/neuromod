import os.path

import numpy as np
from tqdm import tqdm

from metamod.control import LinearNetEq, LinearNetControl, LinearNetConstrainedG
from metamod.tasks import AffineCorrelatedGaussian, MNIST, SemanticTask
from metamod.trainers import two_layer_training
from metamod.networks import LinearNet
# from torch.utils.tensorboard import SummaryWriter

run_name = "contrained_g"
results_path = "../results"

results_dict = {}

dataset_params = {"batch_size": 32,
                  "h_levels": 3}

dataset = SemanticTask(**dataset_params)

model_params = {"learning_rate": 5e-3,
                "hidden_dim": 10,
                "intrinsic_noise": 0.0,
                "reg_coef": 0.0,
                "input_dim": dataset.input_dim,
                "output_dim": dataset.output_dim,
                "W1_0": None,
                "W2_0": None}

model = LinearNet(**model_params)

n_steps = 35
save_weights_every = 20

iters, loss, weights_iter, weights = two_layer_training(model=model, dataset=dataset, n_steps=n_steps, save_weights_every=save_weights_every)

results_dict["iters"] = iters
results_dict["Loss_t_sim"] = loss
results_dict["weights_sim"] = weights
results_dict["weights_iters_sim"] = weights_iter

init_W1 = weights[0][0, ...]
init_W2 = weights[1][0, ...]

init_weights = [init_W1, init_W2]
input_corr, output_corr, input_output_corr, expected_y, expected_x = dataset.get_correlation_matrix()

time_span = np.arange(0, len(iters))*model_params["learning_rate"]
results_dict["time_span"] = time_span

equation_params = {"in_cov": input_corr,
                   "out_cov": output_corr,
                   "in_out_cov": input_output_corr,
                   # "expected_y": expected_y,
                   # "expected_x": expected_x,
                   "init_weights": init_weights,
                   "n_steps": n_steps,
                   "reg_coef": model_params["reg_coef"],
                   "intrinsic_noise": model_params["intrinsic_noise"],
                   "learning_rate": model_params["learning_rate"],
                   "time_constant": 1.0}

solver = LinearNetEq(**equation_params)

control_params = {**equation_params,
                  "control_lower_bound": -0.5,
                  "control_upper_bound": 0.5,
                  "gamma": 0.99,
                  "cost_coef": 0.1,
                  "reward_convertion": 1.0,
                  "init_g": None,
                  "control_lr": 10.0,
                  "degree_of_control": 3,
                  "control_base": "neural_base",
                  "update_first_layer": False,
                  "update_second_layer": True}

control = LinearNetConstrainedG(**control_params)

sim_weights1 = weights[0]
sim_weights2 = weights[1]
print(sim_weights1.shape, sim_weights2.shape)

W1_t, W2_t = solver.get_weights(time_span, get_numpy=True)
Loss_t = solver.get_loss_function(W1_t, W2_t, get_numpy=True)

results_dict["W1_t_eq"] = W1_t
results_dict["W2_t_eq"] = W2_t
results_dict["Loss_t_eq"] = Loss_t

W1_t_control, W2_t_control = control.get_weights(time_span, get_numpy=True)
Loss_t_control = control.get_loss_function(W1_t_control, W2_t_control, get_numpy=True)

results_dict["W1_t_control_init"] = W1_t_control
results_dict["W2_t_control_init"] = W2_t_control
results_dict["Loss_t_control_init"] = Loss_t_control
results_dict["control_signal_init"] = (control.g1_tilda, control.g2_tilda)

flat_W1_t = np.reshape(weights[0], (weights[0].shape[0], -1))
flat_eq_W1_t = np.reshape(W1_t, (W1_t.shape[0], -1))

flat_W2_t = np.reshape(weights[1], (weights[1].shape[0], -1))
flat_eq_W2_t = np.reshape(W2_t, (W2_t.shape[0], -1))

iter_control = 5
control_params["iters_control"] = iter_control
cumulated_reward = []

# writer = SummaryWriter(os.path.join(results_path, run_name))

for i in tqdm(range(iter_control)):
    R = control.train_step(get_numpy=True)
    # print("cumulated reward:", R)
    cumulated_reward.append(R)
cumulated_reward = np.array(cumulated_reward).astype(float)
results_dict["cumulated_reward_opt"] = cumulated_reward