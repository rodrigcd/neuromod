import numpy as np
from bokeh.palettes import Viridis, Category10, Category20


from metamod.tasks import TaskSwitch, AffineCorrelatedGaussian, CompositionOfTasks, SemanticTask
from metamod.trainers import two_layer_training, two_layer_engage_training
from metamod.networks import LinearNet, LinearTaskEngNet

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

engage_coefficients = np.ones((n_steps, len(composition_dataset.datasets)))  # (t, phis)

comp_model = LinearTaskEngNet(**comp_model_params)

iters, comp_loss, weights_iter, weights = two_layer_engage_training(model=comp_model,
                                                                    dataset=composition_dataset,
                                                                    n_steps=n_steps,
                                                                    save_weights_every=save_weights_every,
                                                                    engagement_coefficients=engage_coefficients)
