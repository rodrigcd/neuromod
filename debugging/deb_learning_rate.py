from metamod.tasks import MNIST, SemanticTask
from metamod.networks import LRLinearNet
from metamod.trainers import LR_two_layer_training
from metamod.control import LinearNetEq, LRLinearNetControl
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm


def main(adam_lr):
    run_name = "learning_rate_control"
    results_path = "../results"
    results_dict = {}
    n_steps = 15000
    save_weights_every = 20
    iter_control = 100

    dataset_params = {"batch_size": 32,
                      "h_levels": 4}

    model_params = {"learning_rate": 5e-3,
                    "hidden_dim": 30,
                    "intrinsic_noise": 0.0,
                    "reg_coef": 0.01,
                    "W1_0": None,
                    "W2_0": None}

    dataset_class = SemanticTask

    control_params = {"control_lower_bound": -1.0,
                      "control_upper_bound": 1.0,
                      "gamma": 0.99,
                      "cost_coef": 1e-4,  # 1e-8 for aux, 1e-5 for aux_even_larger
                      "reward_convertion": 1.0,
                      "init_opt_lr": None,
                      "control_lr": adam_lr}  # 0.0005 for cost_coef 0

    dataset = dataset_class(**dataset_params)
    model_params["input_dim"] = dataset.input_dim
    model_params["output_dim"] = dataset.output_dim

    model_params["W1_0"] = np.random.normal(scale=1e-4,
                                            size=(model_params["hidden_dim"], model_params["input_dim"]))
    model_params["W2_0"] = np.random.normal(scale=1e-4,
                                            size=(model_params["output_dim"], model_params["hidden_dim"]))

    # Init neural network
    model = LRLinearNet(**model_params)

    iters, loss, weights_iter, weights = LR_two_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                               save_weights_every=save_weights_every)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter

    # Solving equation
    init_W1 = weights[0][0, ...]
    init_W2 = weights[1][0, ...]

    init_weights = [init_W1, init_W2]
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

    solver = LinearNetEq(**equation_params)

    # Initialize control
    control_params = {**control_params, **copy.deepcopy(equation_params)}
    control = LRLinearNetControl(**control_params)

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
    results_dict["control_signal_init"] = control.opt_lr.detach().cpu().numpy()

    control_params["iters_control"] = iter_control
    cumulated_reward = []
    mean_grad = []

    for i in tqdm(range(iter_control)):
        R, grad = control.train_step(get_numpy=True)
        print(R)
        cumulated_reward.append(R)
        mean_grad.append(np.mean(grad**2))
    cumulated_reward = np.array(cumulated_reward).astype(float)
    results_dict["cumulated_reward_opt"] = cumulated_reward

    W1_t_opt, W2_t_opt = control.get_weights(time_span, get_numpy=True)
    Loss_t_opt = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True)

    f, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(cumulated_reward)
    ax[0, 0].set_title("learning_rate "+str(adam_lr))
    ax[0, 1].plot(control.opt_lr.detach().cpu().numpy())
    ax[0, 1].set_title("control_signal cost"+str(control_params["cost_coef"]))
    ax[1, 0].plot(Loss_t_opt, label="opt")
    ax[1, 0].plot(Loss_t, label="eq")
    ax[1, 0].legend()
    ax[1, 1].plot(mean_grad)
    ax[1, 1].set_title("mean_grad")
    plt.savefig("aux_figures_this_is_it/learning_rate_"+str(adam_lr)+".pdf", bbox_inches='tight')


if __name__ == "__main__":
    # try_lr = [0.00001, 0.00005, 0.0001]
    # try_lr = [0.0005, 0.001, 0.005]
    # try_lr = [0.01, 0.05, 0.1]
    try_lr = [0.5, 1.0]
    for lr in try_lr:
       main(lr)
    # main(0.0005)