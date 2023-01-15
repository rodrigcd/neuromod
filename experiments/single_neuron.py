import copy
import numpy as np
from tqdm import tqdm
import os
from metamod.control import SingleLayerEq, SingleLayerControl
from metamod.trainers import single_layer_training
from metamod.tasks import TwoGaussians
from metamod.networks import SingleLayerNet
from metamod.utils import save_var, get_date_time
import argparse
import sys
from functools import partialmethod


def main(argv):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-name', type=str, default="single_neuron")
    parser.add_argument(
        '--save-path', type=str, default="../results/single_neuron/")
    parser.add_argument(
        '--run-id', type=int, default=0
    )
    args = parser.parse_args(argv)
    args = vars(args)

    run_id = args["run_id"]
    save_path = args["save_path"]
    run_name = args["run_name"] + "_id_" + str(run_id)
    results_path = args["save_path"]

    ##########################################
    # parameters to run ######################
    ##########################################

    n_params = 30
    gammas = 10 ** (np.linspace(-8, 0, n_params, endpoint=True))
    sigmas = np.linspace(1e-5, 5, n_params, endpoint=True)
    betas = np.linspace(1e-5, 2, n_params, endpoint=True)
    reg_coef_values = np.linspace(0, 5, n_params, endpoint=True)
    available_times = [200 + i*50 for i in range(21)]

    print("n_runs", n_params*4 + len(available_times))

    run_vales = {"gamma": 0.99,
                 "sigmas": 1.0,
                 "betas": 0.3,
                 "reg_coef": 0.1,
                 "n_steps": 600}

    if n_params*0 <= run_id < n_params*1:
        param_id = run_id - n_params*0
        run_vales["gamma"] = gammas[param_id]
        print("using gamma", run_vales["gamma"])
    elif n_params*1 <= run_id < n_params*2:
        param_id = run_id - n_params*1
        run_vales["sigmas"] = sigmas[param_id]
        print("using sigmas", run_vales["sigmas"])
    elif n_params*2 <= run_id < n_params*3:
        param_id = run_id - n_params*2
        run_vales["betas"] = betas[param_id]
        print("using betas", run_vales["betas"])
    elif n_params*3 <= run_id < n_params*4:
        param_id = run_id - n_params*3
        run_vales["reg_coef"] = reg_coef_values[param_id]
        print("using reg_coef", run_vales["reg_coef"])
    elif n_params*4 <= run_id < n_params*5:
        param_id = run_id - n_params*4
        run_vales["n_steps"] = available_times[param_id]
        print("using n_steps", run_vales["n_steps"])

    ##########################################

    iter_control = 700
    save_weights_every = 20
    n_steps = run_vales["n_steps"]
    results_dict = {}
    dataset_params = {"mu": 2.0,
                      "batch_size": 128,
                      "std": run_vales["sigmas"]}
    dataset_class = TwoGaussians
    model_params = {"learning_rate": 1e-3,
                    "reg_coef": run_vales["reg_coef"],
                    "intrinsic_noise": 0.0,
                    "W_0": np.zeros((1, 1))}

    control_params = {"control_lower_bound": 0.0,
                      "control_upper_bound": 0.5,
                      "gamma": run_vales["gamma"],
                      "cost_coef": run_vales["betas"],
                      "reward_convertion": 1.0,
                      "init_g": None,
                      "control_lr": 10.0,
                      "square_control_loss": True}

    dataset = dataset_class(**dataset_params)
    model_params["input_dim"] = dataset.input_dim
    model_params["output_dim"] = dataset.output_dim

    model = SingleLayerNet(**model_params)

    iters, loss, weights_iter, weights = single_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                               save_weights_every=save_weights_every)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter

    init_W = weights[0, ...]

    init_weights = init_W
    input_corr, output_corr, input_output_corr, expected_y, expected_x = dataset.get_correlation_matrix()

    time_span = np.arange(0, len(iters)) * model_params["learning_rate"]
    results_dict["time_span"] = time_span

    equation_params = {"in_cov": input_corr,
                       "out_cov": output_corr,
                       "in_out_cov": input_output_corr,
                       "init_weights": init_weights,
                       "n_steps": n_steps,
                       "reg_coef": model_params["reg_coef"],
                       "intrinsic_noise": model_params["intrinsic_noise"],
                       "learning_rate": model_params["learning_rate"],
                       "time_constant": 1.0}

    solver = SingleLayerEq(**equation_params)

    # Initialize control
    control_params = {**control_params, **copy.deepcopy(equation_params)}
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
    control_params["iters_control"] = iter_control

    cumulated_reward = []
    for i in tqdm(range(iter_control)):
        R = control.train_step(get_numpy=True)
        cumulated_reward.append(R)
    cumulated_reward = np.array(cumulated_reward).astype(float)
    results_dict["cumulated_reward_opt"] = cumulated_reward

    W_t_opt = control.get_weights(time_span, get_numpy=True)
    Loss_t_opt = control.get_loss_function(W_t_opt, get_numpy=True)

    results_dict["W_t_control_opt"] = W_t_opt
    results_dict["Loss_t_control_opt"] = Loss_t_opt

    g_tilda = control.g_tilda.detach()
    control_signal = g_tilda
    W_0 = control_params["init_weights"]
    results_dict["control_signal"] = control_signal

    reset_model_params = model_params.copy()
    reset_model_params["W_0"] = W_0

    reset_model = SingleLayerNet(**reset_model_params)

    iters, loss_OPT, weights_iter_OPT, weights_OPT = single_layer_training(model=reset_model,
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
    print("### Saving to", saving_path, "###")
    save_var(results_dict, "results.pkl", results_path=saving_path)
    save_var(params_dict, "params.pkl", results_path=saving_path)


if __name__ == "__main__":
    main(sys.argv[1:])