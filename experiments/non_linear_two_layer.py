import numpy as np
from tqdm import tqdm
import os
from metamod.control import NonLinearNetEq, NonLinearNetControl
from metamod.tasks import AffineCorrelatedGaussian, MultiDimGaussian
from metamod.trainers import two_layer_training
from metamod.networks import NonLinearNet
from metamod.utils import save_var, get_date_time
import argparse
import sys
import copy


def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-name', type=str, default="non_linear")
    parser.add_argument(
        '--save-path', type=str, default="../results/non_linear/")
    parser.add_argument(
        '--datasets', type=str, default="AffineCorrelatedGaussian")
    parser.add_argument(
        '--n-steps', type=int, default=12000)
    parser.add_argument(
        '--iter-control', type=int, default=500)
    args = parser.parse_args(argv)
    args = vars(args)

    run_name = args["run_name"] + "_" + args["datasets"]
    results_path = args["save_path"]

    n_steps = args["n_steps"]
    save_weights_every = 20
    iter_control = args["iter_control"]

    results_dict = {}

    # Init dataset
    batch_size = 2048

    if args["datasets"] == "AffineCorrelatedGaussian":
        print("Using correlated gaussians")
        dataset_params = {"mu_vec": (3.0, 1.0), "sigma_vec": (1.0, 1.0), "dependence_parameter": 0.8,
                          "batch_size": batch_size}
        # Init dataset
        dataset = AffineCorrelatedGaussian(**dataset_params)
    else:
        print("Using 4D gaussians")
        dataset_params = {"mu_vec": np.array((1.0, 3.0, 0.5, 0.5)),
                           "batch_size": batch_size,
                           "max_std": 0.5}
        # Init dataset
        dataset = MultiDimGaussian(**dataset_params)

    model_params = {"learning_rate": 1e-3,
                    "hidden_dim": 8,
                    "intrinsic_noise": 0.00,
                    "reg_coef": 0.01,
                    "W1_0": None,
                    "W2_0": None}

    control_params = {"control_lower_bound": -0.5,
                      "control_upper_bound": 0.5,
                      "gamma": 0.99,
                      "cost_coef": 0.3,
                      "reward_convertion": 1.0,
                      "init_g": None,
                      "control_lr": 5.0}

    print("##### dataset_params #####")
    print(dataset_params)
    print("##### model_params #####")
    print(model_params)
    print("##### control_params #####")
    print(control_params)

    model_params["input_dim"] = dataset.input_dim
    model_params["output_dim"] = dataset.output_dim

    # Init neural network
    model = NonLinearNet(**model_params)

    # Train neural network
    iters, loss, weights_iter, weights = two_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                            save_weights_every=save_weights_every)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter

    # Solving equation
    init_W1 = copy.deepcopy(weights[0][0, ...])
    init_W2 = copy.deepcopy(weights[1][0, ...])

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

    solver = NonLinearNetEq(**equation_params)

    # Initialize control
    control_params = {**control_params, **copy.deepcopy(equation_params)}
    control = NonLinearNetControl(**control_params)

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

    control_params["iters_control"] = iter_control
    cumulated_reward = []

    for i in tqdm(range(iter_control)):
        R = control.train_step(get_numpy=True)
        cumulated_reward.append(R)
    cumulated_reward = np.array(cumulated_reward).astype(float)
    results_dict["cumulated_reward_opt"] = cumulated_reward

    W1_t_opt, W2_t_opt = control.get_weights(time_span, get_numpy=True)
    Loss_t_opt = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True)

    results_dict["W1_t_control_opt"] = W1_t_opt
    results_dict["W2_t_control_opt"] = W2_t_opt
    results_dict["Loss_t_control_opt"] = Loss_t_opt

    g1_tilda = control.g1_tilda
    g2_tilda = control.g2_tilda
    control_signal = (g1_tilda, g2_tilda)
    W1_0, W2_0 = control_params["init_weights"]
    results_dict["control_signal"] = control_signal

    reset_model_params = model_params.copy()
    reset_model_params["W1_0"] = W1_0
    reset_model_params["W2_0"] = W2_0

    reset_model = NonLinearNet(**reset_model_params)

    iters, loss_OPT, weights_iter_OPT, weights_OPT = two_layer_training(model=reset_model,
                                                                        dataset=dataset,
                                                                        n_steps=n_steps,
                                                                        save_weights_every=save_weights_every,
                                                                        control_signal=control_signal)

    results_dict["Loss_t_sim_OPT"] = loss_OPT
    results_dict["weights_sim_OPT"] = weights_OPT
    results_dict["weights_iters_sim_OPT"] = weights_iter_OPT
    results_dict["iters_OPT"] = iters

    equation_params["solver"] = solver
    control_params["control"] = control
    dataset_params["dataset"] = dataset
    model_params["model"] = model
    reset_model_params["model"] = reset_model

    params_dict = {"dataset_params": dataset_params,
                   "model_params": model_params,
                   "equation_params": equation_params,
                   "control_params": control_params,
                   "reset_model_params": reset_model_params}

    time_str = get_date_time()
    saving_path = os.path.join(results_path, run_name + "_" + time_str)
    save_var(results_dict, "results.pkl", results_path=saving_path)
    save_var(params_dict, "params.pkl", results_path=saving_path)


if __name__ == "__main__":
    main(sys.argv[1:])
