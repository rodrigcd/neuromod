import torch
from tqdm import tqdm
from neuromod.tasks import BaseTask
from neuromod.networks import BaseNetwork
import numpy as np


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