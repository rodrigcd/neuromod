import torch
from metamod.tasks import BaseTask
from metamod.networks import FullGradNet
import numpy as np
from tqdm import tqdm


class NonLinearCatEngControl(object):

    def __init__(self, network: FullGradNet, task: BaseTask, inner_loop_iters=6000, outer_loop_iters=50, gamma=0.99,
                 control_lower_bound=-1.0, control_upper_bound=1.0, cost_coef=0.3, reward_convertion=1.0,
                 control_lr=1.0, init_phi=None, type_of_engage="vector"):

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
        self.init_phi = init_phi
        self.control_lr = control_lr
        self.time_span = np.arange(0, self.inner_loop_iters) * self.net.learning_rate
        self.tensor_time_span = torch.tensor(self.time_span, dtype=self.dtype, device=self.device)
        self.dt = self.time_span[1]-self.time_span[0]
        self.type_of_engage = type_of_engage
        self.cost_offset = 0.0 if type_of_engage == "active" else 1.0

        self.engagement_coef = self._get_phi(init_phi)
        self.optimizer = torch.optim.Adam([self.engagement_coef, ], lr=self.control_lr)

    def _get_phi(self, init_phi):
        phi = torch.from_numpy(init_phi).requires_grad_(True).type(self.dtype).to(self.device)
        return phi

    def control_cost(self, get_numpy=False):
        if self.type_of_engage == "vector":
            vector_norm = torch.sum(self.engagement_coef ** 2, dim=-1)
            cost = self.cost_coef * (vector_norm - self.engagement_coef.shape[-1])**2/self.engagement_coef.shape[-1]
        else:
            cost = self.cost_coef*torch.sum((self.engagement_coef - self.cost_offset) ** 2, dim=-1)
        if get_numpy:
            return cost.detach().cpu().numpy()
        else:
            return cost

    def inner_loop(self, t_index):
        x, y = self.task.sample_batch()
        current_loss = self.net.train_step(x, y, phi=self.engagement_coef[t_index, :])
        return current_loss

    def get_loss_function(self, get_numpy=False):
        L_t = []
        for i in range(self.inner_loop_iters):
            loss = self.inner_loop(t_index=i)
            L_t.append(loss)
        L_t = torch.stack(L_t, dim=0)
        if get_numpy:
            return L_t.detach().cpu().numpy()
        else:
            return L_t

    def outer_loop(self, lr=None, show_tqdm=True, get_numpy=False):
        if lr is None:
            lr = self.control_lr

        W1_t_control, W2_t_control = [], []
        L_t = []

        for i in tqdm(range(self.inner_loop_iters), disable=(not show_tqdm)):
            W1_t_control.append(self.net.W1)
            W2_t_control.append(self.net.W2)
            loss = self.inner_loop(t_index=i)
            L_t.append(loss)

        W1_t_control = torch.stack(W1_t_control, dim=0)
        W2_t_control = torch.stack(W2_t_control, dim=0)
        L_t = torch.stack(L_t, dim=0)
        C_t = self.control_cost()

        instant_reward_rate = self.gamma**(self.tensor_time_span)*(self.reward_convertion*L_t+C_t)
        cumulated_R = torch.sum(instant_reward_rate)*self.dt

        # torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        cumulated_R.backward()
        self.optimizer.step()
        cal1 = torch.ones(self.g1.shape).requires_grad_(False).type(self.dtype).to(self.device)
        self.g1_tilda = cal1 + self.g1
        cal1 = torch.ones(self.g2.shape).requires_grad_(False).type(self.dtype).to(self.device)
        self.g2_tilda = cal1 + self.g2

        # for iters in range(inner_loop):
        # self.g1, self.g1_tilda = self._update_g(self.g1, lr=lr)
        # self.g2, self.g2_tilda = self._update_g(self.g2, lr=lr)

        self.net.reset_weights()
        # self.task.set_random_seed(0)

        if get_numpy:
            return cumulated_R.detach().cpu().numpy()
        else:
            return cumulated_R