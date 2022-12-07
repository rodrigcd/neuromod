import numpy as np
from bokeh.palettes import Viridis, Category10, Category20
import matplotlib.pyplot as plt

from metamod.tasks import TaskSwitch, AffineCorrelatedGaussian, CompositionOfTasks, SemanticTask
from metamod.trainers import two_layer_training, two_layer_engage_training
from metamod.networks import LinearNet, LinearTaskEngNet
from metamod.control import LinearNetTaskEngEq, LinearNetTaskEngControl

run_name = "composition_of_tasks"
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

n_steps = 135
save_weights_every = 20
iter_control = 5

iters, baseline_loss, weights_iter, weights = two_layer_training(model=model, dataset=dataset, n_steps=n_steps, save_weights_every=save_weights_every)

results_dict["iters"] = iters
results_dict["Loss_t_sim"] = baseline_loss
results_dict["weights_sim"] = weights
results_dict["weights_iters_sim"] = weights_iter

losses = (baseline_loss, )
colors = (Category10[10][0], )
legends = ("Baseline linear network",)
alphas = (1, )

# s = plot_lines(iters, losses, legends, alphas, colors)
# show(s)

# composition_dataset_params = {"dataset_classes": (SemanticTask,),
#                               "dataset_list_params": (dataset_params,)}
#
# composition_dataset = CompositionOfTasks(**composition_dataset_params)
#
# comp_model_params = {"learning_rate": 5e-3,
#                      "hidden_dim": 10,
#                      "intrinsic_noise": 0.0,
#                      "reg_coef": 0.0,
#                      "input_dim": composition_dataset.input_dim,
#                      "output_dim": composition_dataset.output_dim,
#                      "W1_0": weights[0][0, ...],
#                      "W2_0": weights[1][0, ...],
#                      "task_output_index": composition_dataset.task_output_index}
#
# engage_coefficients = np.ones((n_steps, 1))  # (t, phis)
#
# comp_model = LinearTaskEngNet(**comp_model_params)
#
# iters, comp_loss, weights_iter, weights = two_layer_engage_training(model=comp_model,
#                                                                     dataset=composition_dataset,
#                                                                     n_steps=n_steps,
#                                                                     save_weights_every=save_weights_every,
#                                                                     engagement_coefficients=engage_coefficients)

dataset_params1 = dataset_params.copy()
dataset_params2 = {"batch_size": 32,
                  "h_levels": 4}

composition_dataset_params = {"dataset_classes": (SemanticTask, SemanticTask),
                              "dataset_list_params": (dataset_params1, dataset_params2)}

composition_dataset = CompositionOfTasks(**composition_dataset_params)

comp_model_params = {"learning_rate": 5e-3,
                     "hidden_dim": 10,
                     "intrinsic_noise": 0.0,
                     "reg_coef": 0.0,
                     "input_dim": composition_dataset.input_dim,
                     "output_dim": composition_dataset.output_dim,
                     "W1_0": None,
                     "W2_0": None,
                     "task_output_index": composition_dataset.task_output_index}

engage_coefficients = np.ones((n_steps, len(composition_dataset.datasets)))*0.5  # (t, phis)

comp_model = LinearTaskEngNet(**comp_model_params)

iters, comp_loss, weights_iter, weights = two_layer_engage_training(model=comp_model,
                                                                    dataset=composition_dataset,
                                                                    n_steps=n_steps,
                                                                    save_weights_every=save_weights_every,
                                                                    engagement_coefficients=engage_coefficients)

results_dict["iters"] = iters
results_dict["Loss_t_sim"] = comp_loss
results_dict["weights_sim"] = weights
results_dict["weights_iters_sim"] = weights_iter

init_W1 = weights[0][0, ...]
init_W2 = weights[1][0, ...]

init_weights = [init_W1, init_W2]
input_corr, output_corr, input_output_corr, expected_y, expected_x = composition_dataset.get_correlation_matrix()

time_span = np.arange(0, len(iters))*model_params["learning_rate"]
results_dict["time_span"] = time_span

equation_params = {"in_cov": input_corr,
                   "out_cov": output_corr,
                   "in_out_cov": input_output_corr,
                   "expected_y": expected_y,
                   "expected_x": expected_x,
                   "init_weights": init_weights,
                   "n_steps": n_steps,
                   "reg_coef": model_params["reg_coef"],
                   "intrinsic_noise": model_params["intrinsic_noise"],
                   "learning_rate": model_params["learning_rate"],
                   "time_constant": 1.0,
                   "task_output_index": composition_dataset.task_output_index,
                   "task_input_index": composition_dataset.task_input_index,
                   "engagement_coef": engage_coefficients}


solver = LinearNetTaskEngEq(**equation_params)

# f, ax = plt.subplots(1, 3, figsize=(12, 5))
# ax[0].imshow(solver.overall_in_cov)
# ax[1].imshow(solver.overall_in_out_cov)
# ax[2].imshow(solver.in_out_cov_pertask[0].detach().cpu().numpy())
# plt.show()

control_params = {"control_lower_bound": -1.0,
                  "control_upper_bound": 1.0,
                  "gamma": 0.99,
                  "cost_coef": 0.3,
                  "reward_convertion": 1.0,
                  "init_g": None,
                  "control_lr": 1.0}

control_params = {**control_params, **equation_params}
control = LinearNetTaskEngControl(**control_params)

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
results_dict["engagement_coef"] = control.engagement_coef

control_params["iters_control"] = iter_control
cumulated_reward = []

for i in range(iter_control):
    R = control.train_step(get_numpy=True)
    print(R)
    cumulated_reward.append(R)
cumulated_reward = np.array(cumulated_reward).astype(float)
results_dict["cumulated_reward_opt"] = cumulated_reward

W1_t_opt, W2_t_opt = control.get_weights(time_span, get_numpy=True)
Loss_t_opt = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True)

results_dict["W1_t_control_opt"] = W1_t_opt
results_dict["W2_t_control_opt"] = W2_t_opt
results_dict["Loss_t_control_opt"] = Loss_t_opt

print("debug")












