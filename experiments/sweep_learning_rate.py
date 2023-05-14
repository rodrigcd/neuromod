import copy
import sys

import numpy as np
from tqdm import tqdm
import torch
from metamod.tasks import MNIST, SemanticTask
from metamod.networks import LRLinearNet
from metamod.trainers import LR_two_layer_training
from metamod.control import LinearNetEq, LRLinearNetControl
from metamod.utils import get_date_time, save_var
import os
import argparse


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-name', type=str, default="learning_rate")
    parser.add_argument(
        '--save-path', type=str, default="../results/learning_rate_sweep/")
    parser.add_argument(
        '--run-id', type=int, default=0)
    parser.add_argument(
        '--n-params', type=int, default=10
    )

    args = parser.parse_args(argv)
    args = vars(args)

    args["dataset"] = "Semantic"
    args["optimize_test"] = False
    n_params = args["n_params"]

    gammas = 10 ** (np.linspace(-8, 0, n_params, endpoint=True))
    betas = np.linspace(1e-3, 2, n_params, endpoint=True)

    if args["run_id"] < n_params:
        inner_id = args["run_id"]
        gamma = gammas[inner_id]
        cost_offset = 0.0
        cost_coef = 1.0
    elif n_params <= args["run_id"] < 2*n_params:
        inner_id = args["run_id"] - n_params
        gamma = gammas[inner_id]
        cost_offset = -1.0
        cost_coef = 1.0
    elif 2*n_params <= args["run_id"] < 3*n_params:
        inner_id = args["run_id"] - 2*n_params
        gamma = 1.0
        cost_offset = 0.0
        cost_coef = betas[inner_id]
    elif 3*n_params <= args["run_id"] < 4*n_params:
        inner_id = args["run_id"] - 3*n_params
        gamma = 1.0
        cost_offset = -1.0
        cost_coef = betas[inner_id]

    run_name = args["run_name"] + "_" + args["dataset"]
    results_path = args["save_path"]
    optimize_test = args["optimize_test"]
    save_weights_every = 20
    control_lr = 0.005
    iter_control = 800

    results_dict = {}
    if args["dataset"] == "Semantic":
        dataset_params = {"batch_size": 32,
                          "h_levels": 4}
        dataset_class = SemanticTask
        model_params = {"learning_rate": 5e-3,
                        "hidden_dim": 30,
                        "intrinsic_noise": 0.0,
                        "reg_coef": 0.01,
                        "W1_0": None,
                        "W2_0": None}
        n_steps = 18000

    elif args["dataset"] == "MNIST":
        dataset_params = {"batch_size": 256,
                          "new_shape": (5, 5),
                          "subset": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}
        dataset_class = MNIST
        model_params = {"learning_rate": 5e-2,
                        "hidden_dim": 50,
                        "intrinsic_noise": 0.0,
                        "reg_coef": 0.0,
                        "W1_0": None,
                        "W2_0": None}
        n_steps = 30000

    else:
        print("Invalid dataset")
        return

    control_params = {"control_lower_bound": -1.0,
                      "control_upper_bound": 1.0,
                      "gamma": gamma,
                      "cost_coef": cost_coef,  # 1e-8 for aux, 1e-5 for aux_even_larger
                      "reward_convertion": 1.0,
                      "init_opt_lr": None,
                      "control_lr": control_lr}  # 0.0005 for cost_coef 0

    dataset = dataset_class(**dataset_params)
    model_params["input_dim"] = dataset.input_dim
    model_params["output_dim"] = dataset.output_dim

    if args["dataset"] == "Semantic":
        model_params["W1_0"] = np.random.normal(scale=1e-4,
                                                size=(model_params["hidden_dim"], model_params["input_dim"]))
        model_params["W2_0"] = np.random.normal(scale=1e-4,
                                                size=(model_params["output_dim"], model_params["hidden_dim"]))

    model = LRLinearNet(**model_params)
    iters, loss, test_loss, weights_iter, weights = LR_two_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                                          save_weights_every=save_weights_every,
                                                                          return_test=True)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter
    results_dict["Loss_t_sim_test"] = test_loss

    # Solving equation
    init_W1 = weights[0][0, ...]
    init_W2 = weights[1][0, ...]

    init_weights = [init_W1, init_W2]
    input_corr, output_corr, input_output_corr, expected_y, expected_x = dataset.get_correlation_matrix()
    input_corr_test, output_corr_test, input_output_corr_test, expected_y_test, expected_x_test = dataset.get_correlation_matrix(
        training=False)

    time_span = np.arange(0, len(iters)) * model_params["learning_rate"]
    results_dict["time_span"] = time_span

    equation_params = {"in_cov": input_corr,
                       "out_cov": output_corr,
                       "in_out_cov": input_output_corr,
                       "init_weights": init_weights,
                       "in_cov_test": input_corr_test,
                       "out_cov_test": output_corr_test,
                       "in_out_cov_test": input_output_corr_test,
                       "n_steps": n_steps,
                       "reg_coef": model_params["reg_coef"],
                       "intrinsic_noise": model_params["intrinsic_noise"],
                       "learning_rate": model_params["learning_rate"],
                       "time_constant": 1.0,
                       "cost_offset": cost_offset}

    solver = LinearNetEq(**equation_params)

    # Initialize control
    control_params = {**control_params, **copy.deepcopy(equation_params)}
    control = LRLinearNetControl(**control_params)

    W1_t, W2_t = solver.get_weights(time_span, get_numpy=True)
    Loss_t = solver.get_loss_function(W1_t, W2_t, get_numpy=True)
    Loss_t_test = solver.get_loss_function(W1_t, W2_t, get_numpy=True, use_test=True)

    results_dict["W1_t_eq"] = W1_t
    results_dict["W2_t_eq"] = W2_t
    results_dict["Loss_t_eq"] = Loss_t
    results_dict["Loss_t_eq_test"] = Loss_t_test

    W1_t_control, W2_t_control = control.get_weights(time_span, get_numpy=True)
    Loss_t_control = control.get_loss_function(W1_t_control, W2_t_control, get_numpy=True)
    Loss_t_control_test = control.get_loss_function(W1_t, W2_t, get_numpy=True, use_test=True)

    results_dict["W1_t_control_init"] = W1_t_control
    results_dict["W2_t_control_init"] = W2_t_control
    results_dict["Loss_t_control_init"] = Loss_t_control
    results_dict["Loss_t_control_init_test"] = Loss_t_control_test
    results_dict["control_signal_init"] = control.opt_lr.detach()

    control_params["iters_control"] = iter_control
    cumulated_reward = []
    mean_grad = []

    for i in tqdm(range(iter_control)):
        R, grad = control.train_step(get_numpy=True, eval_on_test=optimize_test)
        cumulated_reward.append(R)
        mean_grad.append(np.mean(grad ** 2))
    cumulated_reward = np.array(cumulated_reward).astype(float)
    results_dict["cumulated_reward_opt"] = cumulated_reward

    W1_t_opt, W2_t_opt = control.get_weights(time_span, get_numpy=True)
    Loss_t_opt = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True)
    Loss_t_opt_test = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True, use_test=True)

    results_dict["W1_t_control_opt"] = W1_t_opt
    results_dict["W2_t_control_opt"] = W2_t_opt
    results_dict["Loss_t_control_opt"] = Loss_t_opt
    results_dict["Loss_t_control_opt_test"] = Loss_t_opt_test

    opt_lr = (control.opt_lr.detach() + torch.ones(control.opt_lr.shape, dtype=model.dtype, device=model.device)) * \
              model_params["learning_rate"]
    control_signal = opt_lr
    W1_0, W2_0 = control_params["init_weights"]
    results_dict["control_signal"] = control.opt_lr.detach()

    reset_model_params = model_params.copy()
    reset_model_params["W1_0"] = W1_0
    reset_model_params["W2_0"] = W2_0

    reset_model = LRLinearNet(**reset_model_params)
    iters, loss_OPT, loss_OPT_test, weights_iter_OPT, weights_OPT = LR_two_layer_training(model=reset_model,
                                                                                          dataset=dataset,
                                                                                          n_steps=n_steps,
                                                                                          save_weights_every=save_weights_every,
                                                                                          opt_lr=opt_lr,
                                                                                          return_test=True)

    results_dict["Loss_t_sim_OPT"] = loss_OPT
    results_dict["Loss_t_sim_OPT_test"] = loss_OPT_test
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
    print("### Saving to", saving_path, "###")
    save_var(results_dict, "results.pkl", results_path=saving_path)
    save_var(params_dict, "params.pkl", results_path=saving_path)


if __name__ == "__main__":
    main(sys.argv[1:])
