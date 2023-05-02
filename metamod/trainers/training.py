import torch
from tqdm import tqdm
from metamod.tasks import BaseTask, MultiTask
from metamod.networks import BaseNetwork, LinearTaskEngNet, LRLinearNet, NetworkSet, LinearNet
import numpy as np
from typing import Union


def single_layer_training(model: BaseNetwork, dataset: BaseTask, n_steps, control_signal=None, save_weights_every=100):

    loss = []
    iters = np.arange(n_steps)
    weights = []
    weights_iter = []
    g_tilda = control_signal

    for t in tqdm(range(n_steps)):
        x, y = dataset.sample_batch()
        if g_tilda is None:
            current_loss = model.train_step(x, y, g_tilda=None)
        else:
            current_loss = model.train_step(x, y, g_tilda=g_tilda[t, :, :])
        loss.append(current_loss.detach().cpu().numpy())
        if t % save_weights_every == 0:
            weights.append(model.W.detach().cpu().numpy())
            weights_iter.append(t)

    return iters, np.array(loss), np.array(weights_iter), np.array(weights)


def two_layer_training(model: BaseNetwork, dataset: BaseTask, n_steps, control_signal=(None, None), save_weights_every=100):

    loss = []
    iters = np.arange(n_steps)
    weights1 = []
    weights2 = []
    weights_iter = []
    g1_tilda = control_signal[0]
    g2_tilda = control_signal[1]

    for t in tqdm(range(n_steps)):
        x, y = dataset.sample_batch()
        if g1_tilda is None:
            current_loss = model.train_step(x, y, g1_tilda=None, g2_tilda=None)
        else:
            current_loss = model.train_step(x, y, g1_tilda=g1_tilda[t, :, :], g2_tilda=g2_tilda[t, :, :])
        loss.append(current_loss.detach().cpu().numpy())
        if t % save_weights_every == 0:
            weights1.append(model.W1.detach().cpu().numpy())
            weights2.append(model.W2.detach().cpu().numpy())
            weights_iter.append(t)

    return iters, np.array(loss), np.array(weights_iter), (np.array(weights1), np.array(weights2))


def two_layer_engage_training(model: LinearTaskEngNet, dataset: BaseTask, n_steps, control_signal=(None, None),
                              save_weights_every=100, engagement_coefficients=None):

    # Engagement coefficient should be a (t, phi) matrix
    loss = []
    iters = np.arange(n_steps)
    weights1 = []
    weights2 = []
    weights_iter = []
    g1_tilda = control_signal[0]
    g2_tilda = control_signal[1]

    for t in tqdm(range(n_steps)):
        if engagement_coefficients is None:
            eng_coef_t = None
        else:
            eng_coef_t = engagement_coefficients[t, :]
        x, y = dataset.sample_batch()
        if g1_tilda is None:
            current_loss = model.train_step(x, y, g1_tilda=None, g2_tilda=None,
                                            engagement_coef=eng_coef_t)
        else:
            current_loss = model.train_step(x, y, g1_tilda=g1_tilda[t, :, :], g2_tilda=g2_tilda[t, :, :],
                                            engagement_coef=eng_coef_t)
        loss.append(current_loss.detach().cpu().numpy())
        if t % save_weights_every == 0:
            weights1.append(model.W1.detach().cpu().numpy())
            weights2.append(model.W2.detach().cpu().numpy())
            weights_iter.append(t)

    return iters, np.array(loss), np.array(weights_iter), (np.array(weights1), np.array(weights2))


def LR_two_layer_training(model: Union[LRLinearNet, LinearNet], dataset: BaseTask, n_steps, opt_lr=None,
                          save_weights_every=100, return_test=False):

    loss = []
    iters = np.arange(n_steps)
    weights1 = []
    weights2 = []
    weights_iter = []
    test_loss = []

    for t in tqdm(range(n_steps)):
        x, y = dataset.sample_batch()
        if return_test:
            x_test, y_test = dataset.sample_batch(training=False)
            current_test_loss = model.loss_function(x_test, y_test)
        if opt_lr is None:
            current_loss = model.train_step(x, y)
        else:
            current_loss = model.train_step(x, y, opt_lr=opt_lr[t])
        loss.append(current_loss.detach().cpu().numpy())
        if return_test:
            test_loss.append(current_test_loss.detach().cpu().numpy())
        if t % save_weights_every == 0:
            weights1.append(model.W1.detach().cpu().numpy())
            weights2.append(model.W2.detach().cpu().numpy())
            weights_iter.append(t)

    if return_test:
        return iters, np.array(loss), np.array(test_loss), np.array(weights_iter), (np.array(weights1), np.array(weights2))
    else:
        return iters, np.array(loss), None, np.array(weights_iter), (np.array(weights1), np.array(weights2))


def set_network_training(model: NetworkSet, dataset: MultiTask, n_steps, save_weights_every=100, return_test=False):

    loss = []
    iters = np.arange(n_steps)
    weights1 = []
    weights2 = []
    weights_iter = []
    test_loss = []

    for i, m in enumerate(model.networks):
        # print(dataset.datasets[i].subsets)
        results = LR_two_layer_training(model=m,
                                        dataset=dataset.datasets[i],
                                        n_steps=n_steps,
                                        save_weights_every=save_weights_every,
                                        return_test=return_test)

        loss.append(results[1])
        test_loss.append(results[2])
        weights_iter.append(results[3])
        weights1.append(results[4][0])
        weights2.append(results[4][1])

    loss = np.stack(loss, axis=0)
    test_loss = np.stack(test_loss, axis=0)
    weights_iter = np.stack(weights_iter, axis=0)
    weights1 = np.stack(weights1, axis=0)
    weights2 = np.stack(weights2, axis=0)

    return iters, loss, test_loss, weights_iter, (np.array(weights1), np.array(weights2))
