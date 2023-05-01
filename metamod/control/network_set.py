import torch
import numpy as np
from .linear_net import LinearNetEq


class NetworkSetEq(object):

    def __init__(self, network_class: LinearNetEq, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 in_out_cov_test=None, in_cov_test=None, out_cov_test=None, optimize_init_weights=False):

        self.n_networks = len(in_out_cov)
        self.networks = []
        for i in range(self.n_networks):
            self.networks.append(network_class(in_out_cov[i], in_cov[i], out_cov[i], init_weights, reg_coef,
                                               intrinsic_noise, learning_rate, n_steps, time_constant,
                                               in_out_cov_test[i], in_cov_test[i], out_cov_test[i],
                                               optimize_init_weights))

    def get_weights(self, time_span, get_numpy=False):
        W1_t = []
        W2_t = []
        for i in range(self.n_networks):
            W1_t_i, W2_t_i = self.networks[i].get_weights(time_span, get_numpy=get_numpy)
            W1_t.append(W1_t_i)
            W2_t.append(W2_t_i)
        if get_numpy:
            W1_t = np.stack(W1_t, axis=0)
            W2_t = np.stack(W2_t, axis=0)
        else:
            W1_t = torch.stack(W1_t, dim=0)
            W2_t = torch.stack(W2_t, dim=0)
        return W1_t, W2_t

    def get_loss_function(self, W1, W2, get_numpy=False, use_test=False):
        Loss_t = []
        for i in range(self.n_networks):
            Loss_t_i = self.networks[i].get_loss_function(W1[i, :, :, :], W2[i, :, :, :],
                                                          get_numpy=get_numpy, use_test=use_test)
            Loss_t.append(Loss_t_i)
        if get_numpy:
            Loss_t = np.stack(Loss_t, axis=0)
        else:
            Loss_t = torch.stack(Loss_t, dim=0)
        return Loss_t


class NetworkSetControl(NetworkSetEq):

    def __init__(self, network_class: LinearNetEq, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 in_out_cov_test=None, in_cov_test=None, out_cov_test=None,
                 control_lower_bound=0.0, control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.W1_0 = torch.tensor(init_weights[0], device=self.device, dtype=self.dtype, requires_grad=True)
        self.W2_0 = torch.tensor(init_weights[1], device=self.device, dtype=self.dtype, requires_grad=True)

        super().__init__(network_class, in_out_cov, in_cov, out_cov, (self.W1_0, self.W2_0), reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant,
                         in_out_cov_test, in_cov_test, out_cov_test, optimize_init_weights=True)

        self.control_upper_bound = control_upper_bound
        self.control_lower_bound = control_lower_bound
        self.gamma = gamma
        self.cost_coef = cost_coef
        self.reward_convertion = reward_convertion
        self.control_lr = control_lr

        self.input_dim = self.W1_0.shape[1]
        self.hidden_dim = self.W1_0.shape[0]
        self.output_dim = self.W2_0.shape[0]
        time_span = np.arange(0, n_steps) * learning_rate
        self.dt = time_span[1]-time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)

        self.optimizer = torch.optim.Adam([self.W1_0, self.W2_0], lr=self.control_lr, amsgrad=True)

    def control_cost(self, get_numpy=False):
        cost = (1/2) * self.cost_coef * (torch.sum(self.W1_0 ** 2)
                                         + torch.sum(self.W2_0 ** 2))
        if get_numpy:
            return cost.detach().cpu().numpy()
        else:
            return cost

    def get_instant_reward_rate(self, L_t=None, C_t=None, get_numpy=True):
        if L_t is None:
            ode_w, W1_t, W2_t, L_t = self.compute_weights_ode()
        else:
            L_t = torch.from_numpy(L_t).type(self.dtype).to(self.device)
        if C_t is None:
            C_t = self.control_cost()
        else:
            C_t = torch.from_numpy(C_t).type(self.dtype).to(self.device)
        instant_reward_rate = -self.reward_convertion * L_t - C_t
        if get_numpy:
            instant_reward_rate = instant_reward_rate.detach().cpu().numpy()
        return instant_reward_rate

    def get_time_span(self, get_numpy=False):
        if get_numpy:
            return self.time_span.detach().cpu().numpy()
        else:
            return self.time_span

    def train_step(self, get_numpy=False, lr=None, eval_on_test=False):
        if lr is None:
            lr = self.control_lr

        W1_t_control, W2_t_control = self.get_weights(self.time_span)
        L_t = self.get_loss_function(W1=W1_t_control, W2=W2_t_control, use_test=eval_on_test)
        C_t = self.control_cost()
        avg_L_t = torch.mean(L_t, dim=0)

        instant_reward_rate = self.gamma**(self.time_span)*(-self.reward_convertion*avg_L_t-C_t)
        cumulated_R = -torch.sum(instant_reward_rate)*self.dt

        self.optimizer.zero_grad()
        cumulated_R.backward()
        self.optimizer.step()
        grad = torch.mean(self.W1_0.grad**2) + torch.mean(self.W2_0.grad**2)

        if get_numpy:
            return cumulated_R.detach().cpu().numpy(), grad.detach().cpu().numpy()
        else:
            return cumulated_R, grad