from metamod.tasks import TaskModulation, MNIST
from metamod.trainers import task_mod_training
from metamod.networks import LinearNet
from metamod.control import LinearNetTaskModEq, LinearNetTaskModControl
# from metamod.utils import plot_lines
import numpy as np


def main():
    run_name = "task_mod"
    results_path = "../results"
    results_dict = {}

    dataset_params1 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (0, 1)}
    dataset_params2 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (7, 1)}
    dataset_params3 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (8, 9)}

    dataset_params = {"dataset_classes": (MNIST, MNIST, MNIST),
                      "dataset_list_params": (dataset_params1, dataset_params2, dataset_params3)}

    dataset_class = TaskModulation

    model_params = {"learning_rate": 5e-3,
                    "hidden_dim": 40,
                    "intrinsic_noise": 0.0,
                    "reg_coef": 0.0,
                    "W1_0": None,
                    "W2_0": None}

    control_lr = 1.0
    iter_control = 10
    n_steps = 130
    save_weights_every = 20

    dataset = dataset_class(**dataset_params)

    comp_model_params = {"learning_rate": 5e-3,
                         "hidden_dim": 20,
                         "intrinsic_noise": 0.0,
                         "reg_coef": 0.0,
                         "input_dim": dataset.input_dim,
                         "output_dim": dataset.output_dim,
                         "W1_0": None,
                         "W2_0": None}

    engage_coefficients = np.ones((n_steps, len(dataset.datasets)))  # (t, phis)
    #engage_coefficients[:, 1] = np.linspace(0, 1, num=engage_coefficients.shape[0])*0.5
    #engage_coefficients[:, 0] = np.linspace(1, 0, num=engage_coefficients.shape[0])*0.5

    comp_model = LinearNet(**comp_model_params)

    iters, loss, weights_iter, weights = task_mod_training(model=comp_model,
                                                           dataset=dataset,
                                                           n_steps=n_steps,
                                                           save_weights_every=save_weights_every,
                                                           engagement_coefficients=engage_coefficients)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter

    init_W1 = weights[0][0, ...]
    init_W2 = weights[1][0, ...]

    init_weights = [init_W1, init_W2]
    input_corr, output_corr, input_output_corr, expected_y, expected_x = dataset.get_correlation_matrix()

    time_span = np.arange(0, len(iters)) * model_params["learning_rate"]
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
                       "time_constant": 1.0}

    solver = LinearNetTaskModEq(**equation_params)

    control_params = {**equation_params,
                      "control_lower_bound": 0.0,
                      "control_upper_bound": 2.0,
                      "gamma": 0.99,
                      "cost_coef": 5.0,
                      "reward_convertion": 1.0,
                      "init_g": None,
                      "control_lr": 1.0}

    control = LinearNetTaskModControl(**control_params)

    W1_t, W2_t = solver.get_weights(time_span, get_numpy=True)
    Loss_t = solver.get_loss_function(W1_t, W2_t, get_numpy=True)

    results_dict["W1_t_eq"] = W1_t
    results_dict["W2_t_eq"] = W2_t
    results_dict["Loss_t_eq"] = Loss_t


if __name__ == "__main__":
    main()