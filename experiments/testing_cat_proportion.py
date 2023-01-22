import torch
from metamod.networks import LinearNet, BaseNetwork
from metamod.utils import ResultsManager
from metamod.tasks import MNIST, MNISTClassMod, BaseTask
from metamod.trainers import two_layer_training
import numpy as np
from tqdm import tqdm
import pickle


def main():
    batch_size = 4048

    results_path = "../results/cat_prop/run_id_1_MNIST_beta_5.0_21-01-2023_01-44-06-936"
    results = ResultsManager(results_path)
    control = results.params["control_params"]["control"]
    class_proportions = control.engagement_coef.detach().cpu().numpy()

    dataset_params = {"batch_size": batch_size,
                      "new_shape": (5, 5),
                      "subset": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}
    balanced_dataset = MNIST(**dataset_params)
    dataset_params["class_proportions"] = class_proportions
    cat_prop_dataset = MNISTClassMod(**dataset_params)
    n_steps = 5000
    save_weights_every = 1000
    model_params = {"learning_rate": 5e-2, "hidden_dim": 50, "intrinsic_noise": 0.0, "reg_coef": 0.0, "W1_0": None,
                    "W2_0": None, "input_dim": balanced_dataset.input_dim, "output_dim": balanced_dataset.output_dim}

    model = LinearNet(**model_params)

    iters, base_loss, weights_iter, weights, base_error_class = custom_standard(model=model, dataset=balanced_dataset,
                                                                           n_steps=n_steps,
                                                                           save_weights_every=save_weights_every)
    print(base_error_class.shape)
    model.reset_weights()

    iters, loss, weights_iter, weights, balanced_loss, control_error_class = custom_training(model=model,
                                                                                             dataset=cat_prop_dataset,
                                                                                             balanced_dataset=balanced_dataset,
                                                                                             n_steps=n_steps,
                                                                                             save_weights_every=save_weights_every)

    results = {"iters": iters,
               "baseline_loss": base_loss,
               "baseline_error_class": base_error_class,
               "curriculum_loss": loss,
               "balanced_loss": balanced_loss,
               "control_error_class": control_error_class}

    pickle.dump(results, open("../results/cat_prop/feedback_curriculum.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def custom_standard(model: BaseNetwork, dataset: BaseTask, n_steps, control_signal=(None, None), save_weights_every=100):

    loss = []
    iters = np.arange(n_steps)
    weights1 = []
    weights2 = []
    weights_iter = []
    error_per_class = []

    for t in tqdm(range(n_steps)):
        x, y = dataset.sample_batch()
        current_loss = model.train_step(x, y)
        error_per_class.append(np.mean((y.T - model.forward(x).detach().cpu().numpy())**2, axis=1))
        loss.append(current_loss.detach().cpu().numpy())
        if t % save_weights_every == 0:
            weights1.append(model.W1.detach().cpu().numpy())
            weights2.append(model.W2.detach().cpu().numpy())
            weights_iter.append(t)

    return iters, np.array(loss), np.array(weights_iter), \
        (np.array(weights1), np.array(weights2)), np.array(error_per_class)


def custom_training(model, dataset, balanced_dataset, n_steps, save_weights_every=100):

    loss = []
    iters = np.arange(n_steps)
    weights1 = []
    weights2 = []
    weights_iter = []
    balanced_loss = []
    error_per_class = []

    for t in tqdm(range(n_steps)):
        x, y = dataset.sample_batch()
        b_x, b_y = balanced_dataset.sample_batch()
        current_loss = model.train_step(x, y)
        error_per_class.append(np.mean((b_y.T - model.forward(b_x).detach().cpu().numpy()) ** 2, axis=1))
        loss.append(current_loss.detach().cpu().numpy())
        balanced_loss.append(model.loss_function(b_x, b_y).detach().cpu().numpy())
        if t % save_weights_every == 0:
            weights1.append(model.W1.detach().cpu().numpy())
            weights2.append(model.W2.detach().cpu().numpy())
            weights_iter.append(t)

    return iters, np.array(loss), np.array(weights_iter),\
        (np.array(weights1), np.array(weights2)), np.array(balanced_loss), np.array(error_per_class)


if __name__ == "__main__":
    main()
