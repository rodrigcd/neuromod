import torch
import numpy as np
from tqdm import tqdm


class NonLinearNetEq(object):

    def __init__(self, in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        self.in_out_cov = torch.from_numpy(in_out_cov).requires_grad_(False).type(self.dtype).to(self.device)
        self.in_cov = torch.from_numpy(in_cov).requires_grad_(False).type(self.dtype).to(self.device)
        self.out_cov = torch.from_numpy(out_cov).requires_grad_(False).type(self.dtype).to(self.device)
        self.expected_x = torch.from_numpy(expected_x).requires_grad_(False).type(self.dtype).to(self.device)
        self.expected_y = torch.from_numpy(expected_y).requires_grad_(False).type(self.dtype).to(self.device)
        self.expected_x = self.expected_x[:, None]
        self.expected_y = self.expected_y[:, None]
        self.reg_coef = reg_coef
        self.time_constant = time_constant
        self.learning_rate = learning_rate
        self.intrinsic_noise = intrinsic_noise
        self.n_steps = n_steps

        self.W1 = torch.from_numpy(init_weights[0]).type(self.dtype).to(self.device)
        self.W2 = torch.from_numpy(init_weights[1]).type(self.dtype).to(self.device)

        self.input_dim = self.W1.shape[1]
        self.hidden_dim = self.W1.shape[0]
        self.output_dim = self.W2.shape[0]

        time_span = np.arange(0, n_steps) * learning_rate
        self.dt = time_span[1]-time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)

    def weight_der(self, t, W1, W2, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        f_term = torch.tanh(W1 @ self.expected_x)
        J, diag_der_f = self.get_der(W1, self.expected_x)

        dW2 = self.expected_y @ f_term.T + self.in_out_cov.T @ J.T + self.expected_y @ self.expected_x.T @ J.T
        dW2 += -W2 @ (f_term @ f_term.T + J @ self.in_cov @ J.T + J @ self.expected_x @ self.expected_x.T @ J.T) - self.reg_coef * W2
        dW2 = dW2/self.time_constant

        dW1 = diag_der_f @ W2.T @ self.in_out_cov.T - diag_der_f @ W2.T @ W2 @ f_term @ self.expected_x.T
        dW1 += -diag_der_f @ W2.T @ W2 @ J @ self.in_cov + diag_der_f @ W2.T @ W2 @ J @ self.expected_x @ self.expected_x.T
        dW1 += -self.reg_coef*W1
        dW1 = dW1/self.time_constant

        return dW1, dW2

    def get_der(self, W1, expected_x):
        der_arg = W1 @ expected_x
        der_f = 1 - torch.tanh(der_arg)**2
        if len(der_f.shape) > 2:
            diag_der_f = torch.diag_embed(der_f[:, :, 0])
        else:
            diag_der_f = torch.diag_embed(der_f[:, 0])
        J = der_f * W1
        return J, diag_der_f

    def get_weights(self, time_span, W1_0=None, W2_0=None, get_numpy=False):
        if W1_0 is None or W2_0 is None:
            W1_0 = self.W1
            W2_0 = self.W2
        W1_t = []
        W2_t = []
        current_W1 = torch.clone(W1_0)
        current_W2 = torch.clone(W2_0)
        W1_t.append(current_W1)
        W2_t.append(current_W2)
        for i, t in enumerate(time_span[:-1]):
            dW1, dW2 = self.weight_der(t, current_W1, current_W2, t_index=i)
            current_W1 = dW1 * self.dt + current_W1
            current_W2 = dW2 * self.dt + current_W2
            W1_t.append(current_W1)
            W2_t.append(current_W2)
        W1_t = torch.stack(W1_t, dim=0)
        W2_t = torch.stack(W2_t, dim=0)
        if get_numpy:
            return W1_t.detach().cpu().numpy(), W2_t.detach().cpu().numpy()
        else:
            return W1_t, W2_t

    def get_loss_function(self, W1, W2, get_numpy=False):

        if isinstance(W1, np.ndarray):
            W1 = torch.from_numpy(W1).type(self.dtype).to(self.device)

        if isinstance(W2, np.ndarray):
            W2 = torch.from_numpy(W2).type(self.dtype).to(self.device)

        f_term = torch.tanh(W1 @ self.expected_x)
        J, diag_der_f = self.get_der(W1, self.expected_x)

        L1_term1 = 1/2*torch.trace(self.out_cov)
        L1_term2 = -1.0*(torch.diagonal(self.expected_y.T @ W2 @ f_term, dim1=-2, dim2=-1).sum(-1)
                         + torch.diagonal(W2 @ J @ self.in_out_cov, dim1=-2, dim2=-1).sum(-1)
                         - torch.diagonal(self.expected_y.T @ W2 @ J @ self.expected_x).sum(-1))
        L1_term3 = 0.5*(torch.diagonal(torch.transpose(f_term, dim0=-1, dim1=-2) @ torch.transpose(W2, dim0=-1, dim1=-2) @ W2 @ f_term).sum(-1)
                        + torch.diagonal(torch.transpose(J, dim0=-1, dim1=-2) @ torch.transpose(W2, dim0=-1, dim1=-2) @ W2 @ J @ self.in_cov, dim1=-2, dim2=-1).sum(-1)
                        - torch.diagonal(torch.transpose(J, dim0=-1, dim1=-2) @ torch.transpose(W2, dim0=-1, dim1=-2) @ W2 @ J @ self.expected_x @ self.expected_x.T, dim1=-2, dim2=-1).sum(-1))

        L2 = (self.reg_coef / 2.0) * (torch.sum(W1 ** 2, (-1, -2)) + torch.sum(W2 ** 2, (-1, -2)))

        L = L1_term1 + L1_term2 + L1_term3 + L2 + 0.5*W2.shape[1]*self.intrinsic_noise**2

        if get_numpy:
            return L.detach().cpu().numpy()
        else:
            return L


class NonLinearNetControl(NonLinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0, control_lower_bound=0.0,
                 control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3, reward_convertion=1.0,
                 control_lr=1e-4):

        super().__init__(in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant)

        self.control_upper_bound = control_upper_bound
        self.control_lower_bound = control_lower_bound
        self.gamma = gamma
        self.cost_coef = cost_coef
        self.reward_convertion = reward_convertion
        self.control_lr = control_lr

        self.g1, self.g1_tilda = self._get_g(init_g, shape=(n_steps, self.W1.shape[0], self.W1.shape[1]))
        self.g2, self.g2_tilda = self._get_g(init_g, shape=(n_steps, self.W2.shape[0], self.W2.shape[1]))

    def _get_g(self, init_g, shape):
        if init_g is None:
            g = torch.normal(mean=0, std=0.01, size=shape,
                                   requires_grad=True, device=self.device, dtype=self.dtype)
            # self.g = torch.rand(size=(len(self.time_span), self.init_weights.shape[0], self.init_weights.shape[1]),
            #                     requires_grad=True, device=self.device, dtype=self.dtype)*0.3
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

    def weight_der(self, t, W1, W2, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        g1_tilda = self.g1_tilda[t_index, :, :]
        g2_tilda = self.g2_tilda[t_index, :, :]

        W1_tilda = W1 * g1_tilda
        W2_tilda = W2 * g2_tilda

        f_term = torch.tanh(W1_tilda @ self.expected_x)
        J, diag_der_f = self.get_der(W1_tilda, self.expected_x)

        dW2 = self.expected_y @ f_term.T + self.in_out_cov.T @ J.T + self.expected_y @ self.expected_x.T @ J.T
        dW2 += -W2_tilda @ (f_term @ f_term.T + J @ self.in_cov @ J.T + J @ self.expected_x @ self.expected_x.T @ J.T)
        dW2 = dW2 * g2_tilda - self.reg_coef * W2
        dW2 = dW2/self.time_constant

        dW1 = diag_der_f @ W2_tilda.T @ self.in_out_cov.T - diag_der_f @ W2_tilda.T @ W2_tilda @ f_term @ self.expected_x.T
        dW1 += -diag_der_f @ W2_tilda.T @ W2_tilda @ J @ self.in_cov + diag_der_f @ W2_tilda.T @ W2_tilda @ J @ self.expected_x @ self.expected_x.T
        dW1 = dW1 * g1_tilda - self.reg_coef * W1
        dW1 = dW1/self.time_constant

        return dW1, dW2

    def get_loss_function(self, W1, W2, get_numpy=False):

        if isinstance(W1, np.ndarray):
            W1 = torch.from_numpy(W1).type(self.dtype).to(self.device)

        if isinstance(W2, np.ndarray):
            W2 = torch.from_numpy(W2).type(self.dtype).to(self.device)

        W1_tilda = W1 * self.g1_tilda
        W2_tilda = W2 * self.g2_tilda

        f_term = torch.tanh(W1_tilda @ self.expected_x)
        J, diag_der_f = self.get_der(W1_tilda, self.expected_x)

        L1_term1 = 1/2*torch.trace(self.out_cov)
        L1_term2 = -1.0*(torch.diagonal(self.expected_y.T @ W2_tilda @ f_term, dim1=-2, dim2=-1).sum(-1)
                         + torch.diagonal(W2_tilda @ J @ self.in_out_cov, dim1=-2, dim2=-1).sum(-1))
        L1_term2 += torch.diagonal(self.expected_y.T @ W2_tilda @ J @ self.expected_x).sum(-1)
        L1_term3 = 0.5*(torch.diagonal(torch.transpose(f_term, dim0=-1, dim1=-2) @ torch.transpose(W2_tilda, dim0=-1, dim1=-2) @ W2_tilda @ f_term).sum(-1)
                        + torch.diagonal(torch.transpose(J, dim0=-1, dim1=-2) @ torch.transpose(W2_tilda, dim0=-1, dim1=-2) @ W2_tilda @ J @ self.in_cov, dim1=-2, dim2=-1).sum(-1)
                        - torch.diagonal(torch.transpose(J, dim0=-1, dim1=-2) @ torch.transpose(W2_tilda, dim0=-1, dim1=-2) @ W2_tilda @ J @ self.expected_x @ self.expected_x.T, dim1=-2, dim2=-1).sum(-1))

        L2 = (self.reg_coef / 2.0) * (torch.sum(W1 ** 2, (-1, -2)) + torch.sum(W2 ** 2, (-1, -2)))

        L = L1_term1 + L1_term2 + L1_term3 + L2 + 0.5*W2.shape[1]*self.intrinsic_noise**2

        if get_numpy:
            return L.detach().cpu().numpy()
        else:
            return L

    def control_cost(self, get_numpy=False):
        cost = torch.exp(
            self.cost_coef * (torch.sum(self.g1 ** 2, dim=(-1, -2)) + torch.sum(self.g2 ** 2, dim=(-1, -2)))) - 1
        if get_numpy:
            return cost.detach().cpu().numpy()
        else:
            return cost

    def train_step(self, get_numpy=False, lr=None, inner_loop=1):
        if lr is None:
            lr = self.control_lr

        W1_t_control, W2_t_control = self.get_weights(self.time_span)
        L_t = self.get_loss_function(W1=W1_t_control, W2=W2_t_control)
        C_t = self.control_cost()

        instant_reward_rate = self.gamma**(self.time_span)*(-self.reward_convertion*L_t-C_t)
        cumulated_R = torch.sum(instant_reward_rate)*self.dt

        cumulated_R.backward()

        # for iters in range(inner_loop):
        self.g1, self.g1_tilda = self._update_g(self.g1, lr=lr)
        self.g2, self.g2_tilda = self._update_g(self.g2, lr=lr)

        if get_numpy:
            return cumulated_R.detach().cpu().numpy()
        else:
            return cumulated_R

    def _update_g(self, g, lr):
        with torch.no_grad():
            g += lr*g.grad
            g.grad = None
            if self.control_upper_bound is None and self.control_lower_bound is not None:
                g.data.clamp_(min=self.control_lower_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is not None:
                g.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is None:
                g.data.clamp_(max=self.control_upper_bound)

        cal1 = torch.ones(g.shape).requires_grad_(False).type(self.dtype).to(self.device)
        g_tilda = cal1 + g
        return g, g_tilda