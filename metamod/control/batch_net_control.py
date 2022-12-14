import numpy as np
import torch
from metamod.networks import LinearNet, NonLinearNet
from metamod.tasks import BaseTask

class BatchNetworkControl(object):

    def __init__(self, network: NonLinearNet, task: BaseTask, inner_loop_iters=6000, outer_loop_iters=50, gamma=0.99,
                 control_lower_bound=-1.0, control_upper_bound=1.0, cost_coef=0.3, reward_convertion=1.0,
                 control_lr = 1.0, init_g = None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        self.net = network
        self.task = task
        self.inner_loop_iters = inner_loop_iters
        self.outer_loop_iters = outer_loop_iters
        self.gamma = gamma
        self.control_lower_bound = control_lower_bound
        self.control_upper_bound = control_upper_bound
        self.cost_coef = cost_coef
        self.reward_convertion = reward_convertion
        self.init_g = init_g
        self.control_lr = control_lr
        time_span = np.arange(0, self.inner_loop_iters) * self.net.learning_rate
        self.dt = time_span[1]-time_span[0]

        self.g1, self.g1_tilda = self._get_g(init_g, shape=(inner_loop_iters, self.net.W1.shape[0], self.net.W1.shape[1]))
        self.g2, self.g2_tilda = self._get_g(init_g, shape=(inner_loop_iters, self.net.W2.shape[0], self.net.W2.shape[1]))

    def _get_g(self, init_g, shape):
        if init_g is None:
            g = torch.normal(mean=0, std=0.01, size=shape,
                             requires_grad=True, device=self.device, dtype=self.dtype)
            if self.control_upper_bound is None and self.control_lower_bound is not None:
                g.data.clamp_(min=self.control_lower_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is not None:
                g.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is None:
                g.data.clamp_(max=self.control_upper_bound)
        else:
            g = torch.from_numpy(init_g).requires_grad_(True).type(self.dtype).to(self.device)
        cal1 = torch.ones(g.shape).requires_grad_(False).type(self.dtype).to(self.device)
        g_tilda = cal1 + g
        return g, g_tilda

    def control_cost(self, get_numpy=False):
        cost = torch.exp(
            self.cost_coef * (torch.sum(self.g1 ** 2, dim=(-1, -2)) + torch.sum(self.g2 ** 2, dim=(-1, -2)))) - 1
        if get_numpy:
            return cost.detach().cpu().numpy()
        else:
            return cost

    def inner_loop(self, t_index):
        x, y = self.task.sample_batch()
        current_loss = self.net.train_step(x, y, g1_tilda=self.g1_tilda[t_index, :, :],
                                           g2_tilda=self.g2_tilda[t_index, :, :])
        return current_loss

    def outer_loop(self, lr=None):
        if lr is None:
            lr = self.control_lr

        W1_t_control, W2_t_control = [], []
        L_t = []

        for i in range(self.inner_loop_iters):
            W1_t_control.append(self.net.W1)
            W2_t_control.append(self.net.W2)
            loss = self.inner_loop(t_index=i)
            L_t.append(loss)

        W1_t_control = torch.stack(W1_t_control, dim=0)
        W2_t_control = torch.stack(W2_t_control, dim=0)
        L_t = torch.stack(L_t, dim=0)
        C_t = self.control_cost()

        instant_reward_rate = self.gamma**(self.time_span)*(-self.reward_convertion*L_t-C_t)
        cumulated_R = torch.sum(instant_reward_rate)*self.dt

        cumulated_R.backward()

        # for iters in range(inner_loop):
        self.g1, self.g1_tilda = self._update_g(self.g1, lr=lr)
        self.g2, self.g2_tilda = self._update_g(self.g2, lr=lr)

        self.net.reset_weights()

        if get_numpy:
            return cumulated_R.detach().cpu().numpy()
        else:
            return cumulated_R