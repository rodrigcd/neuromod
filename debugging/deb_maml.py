from metamod.tasks import MultiTask, MNIST
from metamod.networks import NetworkSet, LinearNet
from metamod.trainers import set_network_training


def main():
    run_name = "deb_maml"
    results_path = "../results"
    results_dict = {}
    n_steps = 150
    save_weights_every = 20
    iter_control = 10
    adam_lr = 0.005

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

    dataset_class = MultiTask

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
                      "gamma": 0.99,
                      "cost_coef": 0,
                      "reward_convertion": 1.0,
                      "init_opt_lr": None,
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


if __name__ == "__main__":
    main()
