from metamod.tasks import MultiTask, MNIST
from metamod.networks import NetworkSet, LinearNet
from metamod.trainers import set_network_training
from metamod.control import NetworkSetEq, LinearNetEq, NetworkSetControl, LastStepSetControl
import numpy as np
import copy
from tqdm import tqdm
from functools import partialmethod
from datetime import datetime


def maml_routine(**kwargs):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    run_name = "deb_maml"
    results_path = "../results"
    results_dict = {}
    n_steps = kwargs["n_steps"]
    eval_steps = kwargs["eval_steps"]
    save_weights_every = 20
    save_grads_every = 200
    iter_control = kwargs["iter_control"]
    last_step = kwargs["last_step"]
    # sample_dataset = kwargs["sample_dataset"]
    adam_lr = 0.005
    control_lr = adam_lr

    dataset_params1 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (0, 1)}
    dataset_params2 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (7, 1)}
    dataset_params3 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (8, 9)}
    dataset_params4 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (3, 8)}
    dataset_params5 = {"batch_size": 256,
                       "new_shape": (5, 5),
                       "subset": (5, 3)}
    dataset_params = {"dataset_classes": (MNIST, MNIST, MNIST, MNIST, MNIST),
                      "dataset_list_params": (dataset_params1, dataset_params2, dataset_params3,
                                              dataset_params4, dataset_params5)}

    dataset_class = MultiTask
    optimize_test = False

    network_params = {"learning_rate": 5e-3,
                      "hidden_dim": 40,
                      "intrinsic_noise": 0.0,
                      "reg_coef": 0.0,
                      "W1_0": None,
                      "W2_0": None}

    network_class = LinearNet
    network_copies = len(dataset_params["dataset_classes"])

    model_params = {"network_class": network_class,
                    "network_params": network_params,
                    "n_copies": network_copies}

    control_params = {"control_lower_bound": -1.0,
                      "control_upper_bound": 1.0,
                      "gamma": 1.0,
                      "cost_coef": 0,
                      "reward_convertion": 1.0,
                      "control_lr": control_lr}

    dataset = dataset_class(**dataset_params)
    model_params["network_params"]["input_dim"] = dataset.input_dim
    model_params["network_params"]["output_dim"] = dataset.output_dim

    # Init neural network
    model = NetworkSet(**model_params)

    iters, loss, test_loss, weights_iter, weights = set_network_training(model=model, dataset=dataset, n_steps=n_steps,
                                                                         save_weights_every=save_weights_every,
                                                                         return_test=True)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter
    results_dict["Loss_t_sim_test"] = test_loss
    results_dict["avg_Loss_t_sim"] = np.mean(loss, axis=0)
    results_dict["avg_Loss_t_sim_test"] = np.mean(test_loss, axis=0)

    # Solving equation
    init_W1 = weights[0][0, 0, ...]
    init_W2 = weights[1][0, 0, ...]

    init_weights = [init_W1, init_W2]
    input_corr, output_corr, input_output_corr, expected_y, expected_x = dataset.get_correlation_matrix()
    input_corr_test, output_corr_test, input_output_corr_test, expected_y_test, expected_x_test = dataset.get_correlation_matrix(
        training=False)

    time_span = np.arange(0, len(iters)) * model_params["network_params"]["learning_rate"]
    eval_time_span = np.arange(0, eval_steps) * model_params["network_params"]["learning_rate"]
    results_dict["time_span"] = time_span
    results_dict["eval_time_span"] = eval_time_span

    equation_params = {"network_class": LinearNetEq,
                       "in_cov": input_corr,
                       "out_cov": output_corr,
                       "in_out_cov": input_output_corr,
                       "init_weights": init_weights,
                       "in_cov_test": input_corr_test,
                       "out_cov_test": output_corr_test,
                       "in_out_cov_test": input_output_corr_test,
                       "n_steps": n_steps,
                       "reg_coef": model_params["network_params"]["reg_coef"],
                       "intrinsic_noise": model_params["network_params"]["intrinsic_noise"],
                       "learning_rate": model_params["network_params"]["learning_rate"],
                       "time_constant": 1.0}

    #if sample_dataset:

    solver = NetworkSetEq(**equation_params)

    W1_t, W2_t = solver.get_weights(time_span, get_numpy=True)
    Loss_t = solver.get_loss_function(W1_t, W2_t, get_numpy=True)
    Loss_t_test = solver.get_loss_function(W1_t, W2_t, get_numpy=True, use_test=True)

    results_dict["W1_t_eq"] = W1_t
    results_dict["W2_t_eq"] = W2_t
    results_dict["Loss_t_eq"] = Loss_t
    results_dict["Loss_t_eq_test"] = Loss_t_test

    control_params = {**control_params, **copy.deepcopy(equation_params)}
    if last_step:
        control = LastStepSetControl(**control_params)
        print("MAML Last step mask")
    else:
        control = NetworkSetControl(**control_params)
        print("MAML First step mask")

    W1_t_control, W2_t_control = control.get_weights(time_span, get_numpy=True)
    Loss_t_control = control.get_loss_function(W1_t_control, W2_t_control, get_numpy=True)
    Loss_t_control_test = control.get_loss_function(W1_t, W2_t, get_numpy=True, use_test=True)

    results_dict["W1_t_control_init"] = W1_t_control
    results_dict["W2_t_control_init"] = W2_t_control
    results_dict["Loss_t_control_init"] = Loss_t_control
    results_dict["Loss_t_control_init_test"] = Loss_t_control_test
    results_dict["control_signal_init"] = (control.W1_0, control.W2_0)

    control_params["iters_control"] = iter_control
    cumulated_reward = []
    W1_control_grad = []
    W2_control_grad = []

    start_time = datetime.now()
    for i in tqdm(range(iter_control)):
        R, W1_grad, W2_grad = control.train_step(get_numpy=True, eval_on_test=optimize_test)
        cumulated_reward.append(R)
        # if i % save_grads_every == 0 or i == iter_control - 1:
        #     W1_control_grad.append(W1_grad)
        #     W2_control_grad.append(W2_grad)
    end_time = datetime.now()
    results_dict["optimization_time"] = (end_time - start_time).total_seconds()

    cumulated_reward = np.array(cumulated_reward).astype(float)
    results_dict["cumulated_reward_opt"] = cumulated_reward
    results_dict["W1_grad"] = W1_control_grad
    results_dict["W2_grad"] = W2_control_grad

    W1_t_opt, W2_t_opt = control.get_weights(eval_time_span, get_numpy=True)
    Loss_t_opt = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True)
    Loss_t_opt_test = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True, use_test=True)

    results_dict["W1_t_control_opt"] = W1_t_opt
    results_dict["W2_t_control_opt"] = W2_t_opt
    results_dict["Loss_t_control_opt"] = Loss_t_opt
    results_dict["Loss_t_control_opt_test"] = Loss_t_opt_test

    return results_dict


if __name__ == "__main__":
    iter_control = 100
    eval_steps = 100
    steps = 20
    results = maml_routine(n_steps=steps,
                           eval_steps=eval_steps,
                           iter_control=iter_control,
                           last_step=False)
    results = maml_routine(n_steps=steps,
                           eval_steps=eval_steps,
                           iter_control=iter_control,
                           last_step=True)
    print("done!")