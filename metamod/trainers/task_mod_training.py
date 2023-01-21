from metamod.networks import LinearNet
from metamod.tasks import TaskModulation
import numpy as np
from tqdm import tqdm
from scipy.special import softmax


def task_mod_training(model: LinearNet, dataset: TaskModulation, n_steps,
                      save_weights_every=100, engagement_coefficients=None):

    # Engagement coefficient should be a (t, phi) matrix
    loss = []
    iters = np.arange(n_steps)
    weights1 = []
    weights2 = []
    weights_iter = []

    for t in tqdm(range(n_steps)):
        if engagement_coefficients is None:
            eng_coef_t = None
        else:
            eng_coef_t = engagement_coefficients[t, :]

        s_coef = softmax(eng_coef_t)

        x, y = dataset.sample_batch(sigmoid_coef=s_coef)
        current_loss = model.train_step(x, y)
        loss.append(current_loss.detach().cpu().numpy())
        if t % save_weights_every == 0:
            weights1.append(model.W1.detach().cpu().numpy())
            weights2.append(model.W2.detach().cpu().numpy())
            weights_iter.append(t)

    return iters, np.array(loss), np.array(weights_iter), (np.array(weights1), np.array(weights2))