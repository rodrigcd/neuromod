import numpy as np

from metamod.control import LinearNetControl
import torch


class LinearNetConstrainedG(LinearNetControl):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 control_lower_bound=0.0, control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4, degree_of_control="full", control_base="canonical"):

        self.degree_of_control = degree_of_control
        self.control_base = control_base
        self.g_basis = []
        self.basis_nus = []

        super().__init__(in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant,
                         control_lower_bound, control_upper_bound, init_g, gamma, cost_coef,
                         reward_convertion, control_lr)

    def _get_g(self, init_g, shape):
        if self.degree_of_control == "full":
            self.degree_of_control = shape[1] * shape[2]
        nus = torch.normal(mean=0, std=0.001, size=(shape[0], 1, 1, self.degree_of_control)).requires_grad_(True).type(self.dtype).to(self.device)
        g_base = self._generate_base(shape)

        if self.control_upper_bound is None and self.control_lower_bound is not None:
            nus.data.clamp_(min=self.control_lower_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is not None:
            nus.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is None:
            nus.data.clamp_(max=self.control_upper_bound)

        self.g_basis.append(g_base)
        self.basis_nus.append(nus)

        g_per_degree = nus*g_base
        g = torch.sum(g_per_degree, dim=3)
        cal1 = torch.ones(g.shape).requires_grad_(False).type(self.dtype).to(self.device)
        g_tilda = cal1 + g
        return g, g_tilda

    def _update_g(self, g, lr):
        with torch.no_grad():
            self.basis_nus[self.update_layer] += lr*self.basis_nus[self.update_layer].grad
            self.basis_nus[self.update_layer].grad = None

            if self.control_upper_bound is None and self.control_lower_bound is not None:
                self.basis_nus[self.update_layer].data.clamp_(min=self.control_lower_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is not None:
                self.basis_nus[self.update_layer].data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is None:
                self.basis_nus[self.update_layer].data.clamp_(max=self.control_upper_bound)

            g_per_degree = self.basis_nus[self.update_layer] * self.g_basis[self.update_layer]
            g = torch.sum(g_per_degree, dim=3)
            cal1 = torch.ones(g.shape).requires_grad_(False).type(self.dtype).to(self.device)
            g_tilda = cal1 + g
            return g, g_tilda

    def _generate_base(self, shape):
        if self.control_base == "canonical":
            g_s = torch.zeros(size=(shape[1]*shape[2], self.degree_of_control)).requires_grad_(False).type(self.dtype).to(self.device)
            indexes = np.random.choice(np.arange(shape[1]*shape[2]), size=self.degree_of_control, replace=False).astype(int)
            for i, ind in enumerate(indexes):
                g_s[ind, i] = 1
            g_s = torch.reshape(g_s, shape=(1, shape[1], shape[2], self.degree_of_control))
            return g_s

        elif self.control_base == "frequency":
            pass