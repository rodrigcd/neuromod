from neuromod.control import NonLinearEq, NonLinearControl
from neuromod.tasks import AffineCorrelatedGaussian
from neuromod.trainers import two_layer_training
from neuromod.networks import NonLinearNet

def main():
    dataset_params = {"mu_vec": (3.0, 1.0),
                      "batch_size": 1024,
                      "dependence_parameter": 0.8,
                      "sigma_vec": (1.0, 1.0)}

    dataset = AffineCorrelatedGaussian(**dataset_params)

    model_params = {"learning_rate": 1e-3,
                    "hidden_dim": 4,
                    "intrinsic_noise": 0.05,
                    "reg_coef": 0.0,
                    "input_dim": dataset.input_dim,
                    "output_dim": dataset.output_dim,
                    "W1_0": None,
                    "W2_0": None}

    model = NonLinearNet(**model_params)

    control_params = {"control_lower_bound": -0.5,
                      "control_upper_bound": 0.5,
                      "gamma": 0.99,
                      "cost_coef": 0.3,
                      "reward_convertion": 1.0,
                      "init_g": None,
                      "control_lr": 10.0}

    n_steps = 6000
    save_weights_every = 20

    iters, loss, weights_iter, weights = two_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                            save_weights_every=save_weights_every)


if __name__ == "__main__":
    main()