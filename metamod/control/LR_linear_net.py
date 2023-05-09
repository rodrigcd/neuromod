import torch
import numpy as np
import matplotlib.pyplot as plt
from .linear_net import LinearNetEq, LinearNetControl


class LRLinearNetControl(LinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 in_out_cov_test=None, in_cov_test=None, out_cov_test=None,
                 control_lower_bound=-0.5, control_upper_bound=0.5, init_opt_lr=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4, cost_offset=0.0):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        super().__init__(in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant,
                         in_out_cov_test, in_cov_test, out_cov_test)

        self.control_upper_bound = control_upper_bound
        self.control_lower_bound = control_lower_bound
        self.gamma = gamma
        self.cost_coef = cost_coef
        self.reward_convertion = reward_convertion
        self.control_lr = control_lr
        self.base_lr = self.learning_rate
        self.cost_offset = cost_offset

        self.W1 = torch.from_numpy(init_weights[0]).type(self.dtype).to(self.device)
        self.W2 = torch.from_numpy(init_weights[1]).type(self.dtype).to(self.device)
        self.input_dim = self.W1.shape[1]
        self.hidden_dim = self.W1.shape[0]
        self.output_dim = self.W2.shape[0]
        time_span = np.arange(0, n_steps) * learning_rate
        self.dt = time_span[1]-time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)
        self.opt_lr, self.opt_lr_tilda = self._get_opt_lr(init_opt_lr, time_span.shape)
        self.optimizer = torch.optim.Adam([self.opt_lr], lr=self.control_lr, amsgrad=True)

    def _get_opt_lr(self, init_opt_lr, shape):
        if init_opt_lr is None:
            # opt_lr = torch.normal(mean=0, std=0.01, size=shape,
            #                       requires_grad=True, device=self.device, dtype=self.dtype)
            opt_lr = torch.zeros(shape, requires_grad=True, dtype=self.dtype, device=self.device)#.requires_grad_(True).type(self.dtype).to(self.device)
            if self.control_upper_bound is None and self.control_lower_bound is not None:
                opt_lr.data.clamp_(min=self.control_lower_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is not None:
                opt_lr.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is None:
                opt_lr.data.clamp_(max=self.control_upper_bound)
        else:
            opt_lr = torch.from_numpy(init_opt_lr).requires_grad_(True).type(self.dtype).to(self.device)
        cal1 = torch.ones(opt_lr.shape).requires_grad_(False).type(self.dtype).to(self.device)
        opt_lr_tilda = cal1 + opt_lr
        return opt_lr, opt_lr_tilda

    def weight_der(self, t, W1, W2, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        dW1 = self.opt_lr_tilda[t_index]*(W2.T @ (self.in_out_cov.T - W2 @ W1 @ self.in_cov) - self.reg_coef * W1)/self.time_constant
        dW2 = self.opt_lr_tilda[t_index]*((self.in_out_cov.T - W2 @ W1 @ self.in_cov) @ W1.T - self.reg_coef * W2)/self.time_constant
        return dW1, dW2

    def control_cost(self, get_numpy=False):
        cost = self.cost_coef * (self.opt_lr - self.cost_offset) ** 2
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

        instant_reward_rate = self.gamma**(self.time_span)*(-self.reward_convertion*L_t-C_t)
        # Minimizing for the optimizer
        cumulated_R = -torch.sum(instant_reward_rate)*self.dt

        self.optimizer.zero_grad()
        cumulated_R.backward()
        self.optimizer.step()
        self.opt_lr, self.opt_lr_tilda = self._bound_opt_lr(self.opt_lr)
        # self.opt_lr, self.opt_lr_tilda = self._update_opt_lr(self.opt_lr, lr=lr)

        if get_numpy:
            return cumulated_R.detach().cpu().numpy(), self.opt_lr.grad.detach().cpu().numpy()
        else:
            return cumulated_R, self.opt_lr.grad

    def _update_opt_lr(self, opt_lr, lr):
        print(torch.mean(opt_lr.grad), torch.mean(opt_lr))
        with torch.no_grad():
            opt_lr += lr*opt_lr.grad
            opt_lr.grad = None
            if self.control_upper_bound is None and self.control_lower_bound is not None:
                opt_lr.data.clamp_(min=self.control_lower_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is not None:
                opt_lr.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is None:
                opt_lr.data.clamp_(max=self.control_upper_bound)

        cal1 = torch.ones(opt_lr.shape).requires_grad_(False).type(self.dtype).to(self.device)
        opt_lr_tilda = cal1 + opt_lr
        return opt_lr, opt_lr_tilda

    def _bound_opt_lr(self, opt_lr):
        if self.control_upper_bound is None and self.control_lower_bound is not None:
            opt_lr.data.clamp_(min=self.control_lower_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is not None:
            opt_lr.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is None:
            opt_lr.data.clamp_(max=self.control_upper_bound)

        cal1 = torch.ones(opt_lr.shape).requires_grad_(False).type(self.dtype).to(self.device)
        opt_lr_tilda = cal1 + opt_lr
        return opt_lr, opt_lr_tilda