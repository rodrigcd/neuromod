import numpy as np
from metamod.control import SingleLayerEq, SingleLayerControl
from metamod.tasks import AffineCorrelatedGaussian
from metamod.trainers import single_layer_training
from metamod.networks import SingleLayerNet
# from metamod.utils import plot_lines, plot_weight_ev, check_dir, save_var, get_date_time


def main():
    run_name = "testing_single_layer"
    results_path = "../results"
    results_dict = {}

    dataset_params = {"mu_vec": (3.0, 1.0),
                      "batch_size": 256,
                      "dependence_parameter": 0.8,
                      "sigma_vec": (1.0, 1.0)}

    dataset = AffineCorrelatedGaussian(**dataset_params)

    model_params = {"learning_rate": 1e-3,
                    "reg_coef": 0.05,
                    "intrinsic_noise": 0.0,
                    "input_dim": dataset.input_dim,
                    "output_dim": dataset.output_dim,
                    "W_0": None}

    model = SingleLayerNet(**model_params)

    n_steps = 50
    save_weights_every = 20

    iters, loss, weights_iter, weights = single_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                               save_weights_every=save_weights_every)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter

    init_W = weights[0, ...]

    init_weights = init_W
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

    solver = SingleLayerEq(**equation_params)
    control_params = {**equation_params,
                      "control_lower_bound": -0.5,
                      "control_upper_bound": 0.5,
                      "gamma": 0.99,
                      "cost_coef": 0.3,
                      "reward_convertion": 1.0,
                      "init_g": None,
                      "control_lr": 10.0}

    control = SingleLayerControl(**control_params)
    sim_weights = weights
    print(sim_weights.shape, control.g_tilda.shape)

    W_t = solver.get_weights(time_span, get_numpy=True)
    Loss_t = solver.get_loss_function(W_t, get_numpy=True)

    results_dict["W_t_eq"] = W_t
    results_dict["Loss_t_eq"] = Loss_t

    W_t_control = control.get_weights(time_span, get_numpy=True)
    Loss_t_control = control.get_loss_function(W_t_control, get_numpy=True)

    results_dict["W_t_control_init"] = W_t_control
    results_dict["Loss_t_control_init"] = Loss_t_control
    results_dict["control_signal_init"] = control.g_tilda


if __name__ == "__main__":
    main()
