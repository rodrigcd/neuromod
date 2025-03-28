from metamod.tasks import MultiTask, MNIST
from metamod.networks import NetworkSet, LinearNet
from metamod.trainers import set_network_training
from metamod.control import NetworkSetEq, LinearNetEq, NetworkSetControl
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    run_name = "deb_maml"
    results_path = "../results"
    results_dict = {}
    n_steps = 300
    save_weights_every = 20
    iter_control = 1500
    adam_lr = 0.005
    batch_size = 16

    dataset_params1 = {"batch_size": batch_size,
                       "new_shape": (5, 5),
                       "subset": (0, 1)}
    dataset_params2 = {"batch_size": batch_size,
                       "new_shape": (5, 5),
                       "subset": (7, 1)}
    dataset_params3 = {"batch_size": batch_size,
                       "new_shape": (5, 5),
                       "subset": (8, 9)}
    dataset_params = {"dataset_classes": (MNIST, MNIST, MNIST),
                      "dataset_list_params": (dataset_params1, dataset_params2, dataset_params3)}

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

    control_lr = adam_lr

    control_params = {"control_lower_bound": -1.0,
                      "control_upper_bound": 1.0,
                      "gamma": 1.0,
                      "cost_coef": 0,
                      "reward_convertion": 1.0,
                      "control_lr": adam_lr}

    dataset = dataset_class(**dataset_params)
    model_params["network_params"]["input_dim"] = dataset.input_dim
    model_params["network_params"]["output_dim"] = dataset.output_dim

    # Init neural network
    model = NetworkSet(**model_params)

    # Train neural network
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
    results_dict["time_span"] = time_span

    equation_params = {"network_class": LinearNetEq,
                       "datasets": dataset.datasets,  # It this is not None, it will sample batches for the eqs
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

    approx_solver = NetworkSetEq(**equation_params)

    start = datetime.now()
    W1_t, W2_t = approx_solver.get_weights(time_span, get_numpy=True)
    Loss_t = approx_solver.get_loss_function(W1_t, W2_t, get_numpy=True)
    Loss_t_test = approx_solver.get_loss_function(W1_t, W2_t, get_numpy=True, use_test=True)
    end = datetime.now()
    print("approx_solver time: ", end - start)

    real_equation_params = copy.deepcopy(equation_params)
    real_equation_params["datasets"] = None
    real_solver = NetworkSetEq(**real_equation_params)

    start = datetime.now()
    W1_t_real, W2_t_real = real_solver.get_weights(time_span, get_numpy=True)
    Loss_t_real = real_solver.get_loss_function(W1_t_real, W2_t_real, get_numpy=True)
    Loss_t_real_test = real_solver.get_loss_function(W1_t_real, W2_t_real, get_numpy=True, use_test=True)
    end = datetime.now()
    print("real_solver time: ", end - start)

    #plt.plot(np.mean(Loss_t, axis=0), label="approx_solver")
    #plt.plot(np.mean(Loss_t_real, axis=0), label="real_solver")
    #plt.legend()
    #plt.show()

    results_dict["W1_t_eq"] = W1_t
    results_dict["W2_t_eq"] = W2_t
    results_dict["Loss_t_eq"] = Loss_t
    results_dict["Loss_t_eq_test"] = Loss_t_test

    # Initialize control
    control_params = {**control_params, **copy.deepcopy(equation_params)}
    control = NetworkSetControl(**control_params)

    real_control_params = {**control_params, **copy.deepcopy(real_equation_params)}
    real_control = NetworkSetControl(**real_control_params)

    control_params["iters_control"] = iter_control
    cumulated_reward = []
    mean_grad = []
    for i in tqdm(range(iter_control)):
        R, grad, _ = control.train_step(get_numpy=True, eval_on_test=optimize_test)
        cumulated_reward.append(R)

    real_cumulated_reward = []
    mean_grad = []
    for i in tqdm(range(iter_control)):
        R, grad, _ = real_control.train_step(get_numpy=True, eval_on_test=optimize_test)
        real_cumulated_reward.append(R)

    plt.plot(cumulated_reward, label="approx_solver")
    plt.plot(real_cumulated_reward, label="real_solver")
    plt.legend()
    plt.show()

    # mean_grad.append(grad)
    # cumulated_reward = np.array(cumulated_reward).astype(float)
    # results_dict["cumulated_reward_opt"] = cumulated_reward
    #
    # f, ax = plt.subplots(1, 2, figsize=(12, 5))
    # ax[0].plot(cumulated_reward)
    # ax[0].set_title("MAML loss")
    # ax[1].plot(mean_grad)
    # ax[1].set_title("mean_grad")
    # plt.show()
    #
    # print("debug")


if __name__ == "__main__":
    main()
