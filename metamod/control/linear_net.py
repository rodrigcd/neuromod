import torch
import numpy as np


class LinearNetEq(object):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        self.in_out_cov = torch.from_numpy(in_out_cov).requires_grad_(False).type(self.dtype).to(self.device)
        self.in_cov = torch.from_numpy(in_cov).requires_grad_(False).type(self.dtype).to(self.device)
        self.out_cov = torch.from_numpy(out_cov).requires_grad_(False).type(self.dtype).to(self.device)
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
        self.dt = time_span[1] - time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)

    def weight_der(self, t, W1, W2, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        dW1 = (W2.T @ (self.in_out_cov.T - W2 @ W1 @ self.in_cov) - self.reg_coef * W1)/self.time_constant
        dW2 = ((self.in_out_cov.T - W2 @ W1 @ self.in_cov) @ W1.T - self.reg_coef * W2)/self.time_constant

        return dW1, dW2

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
        W_t = W2 @ W1
        if isinstance(W_t, np.ndarray):
            W_t = torch.from_numpy(W_t).type(self.dtype).to(self.device)
            W2 = torch.from_numpy(W2).type(self.dtype).to(self.device)
            W1 = torch.from_numpy(W1).type(self.dtype).to(self.device)
        L1 = 0.5*(torch.trace(self.out_cov) - torch.diagonal(2*self.in_out_cov @ W_t, dim1=-2, dim2=-1).sum(-1)
                          + torch.diagonal(self.in_cov @ torch.transpose(W_t, dim0=-1, dim1=-2) @ W_t,
                                           dim1=-2, dim2=-1).sum(-1)) + 0.5*W_t.shape[1]*self.intrinsic_noise**2
        L2 = (self.reg_coef/2.0)*(torch.sum(W1**2, (-1, -2)) + torch.sum(W2**2, (-1, -2)))
        L = L1 + L2
        if get_numpy:
            return L.detach().cpu().numpy()
        else:
            return L


class LinearNetControl(LinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 control_lower_bound=0.0, control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        super().__init__(in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant)

        self.control_upper_bound = control_upper_bound
        self.control_lower_bound = control_lower_bound
        self.gamma = gamma
        self.cost_coef = cost_coef
        self.reward_convertion = reward_convertion
        self.control_lr = control_lr

        self.W1 = torch.from_numpy(init_weights[0]).type(self.dtype).to(self.device)
        self.W2 = torch.from_numpy(init_weights[1]).type(self.dtype).to(self.device)
        self.input_dim = self.W1.shape[1]
        self.hidden_dim = self.W1.shape[0]
        self.output_dim = self.W2.shape[0]
        time_span = np.arange(0, n_steps) * learning_rate
        self.dt = time_span[1]-time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)

        self.update_layer = 0  # Kind of a hack for other code :(
        self.g1, self.g1_tilda = self._get_g(init_g, shape=(n_steps, self.W1.shape[0], self.W1.shape[1]))
        self.update_layer = 1
        self.g2, self.g2_tilda = self._get_g(init_g, shape=(n_steps, self.W2.shape[0], self.W2.shape[1]))

    def _get_g(self, init_g, shape):
        if init_g is None:
            g = torch.normal(mean=0, std=0.001, size=shape,
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

        dW1 = ((W2_tilda.T @ (self.in_out_cov.T - W2_tilda @ W1_tilda @ self.in_cov)) * g1_tilda - self.reg_coef * W1)/self.time_constant
        dW2 = (((self.in_out_cov.T - W2_tilda @ W1_tilda @ self.in_cov) @ W1_tilda.T) * g2_tilda - self.reg_coef * W2)/self.time_constant

        return dW1, dW2

    def get_loss_function(self, W1, W2, get_numpy=False):
        if isinstance(W1, np.ndarray):
            W2 = torch.from_numpy(W2).type(self.dtype).to(self.device)
            W1 = torch.from_numpy(W1).type(self.dtype).to(self.device)

        W_t = (W2 * self.g2_tilda) @ (W1 * self.g1_tilda)
        L1 = 0.5*(torch.trace(self.out_cov) - torch.diagonal(2*self.in_out_cov @ W_t, dim1=-2, dim2=-1).sum(-1)
                          + torch.diagonal(self.in_cov @ torch.transpose(W_t, dim0=-1, dim1=-2) @ W_t,
                                           dim1=-2, dim2=-1).sum(-1)) + 0.5*W_t.shape[1]*self.intrinsic_noise**2
        L2 = (self.reg_coef/2.0)*(torch.sum(W1**2, (-1, -2)) + torch.sum(W2**2, (-1, -2)))
        L = L1 + L2
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

    def train_step(self, get_numpy=False, lr=None):
        if lr is None:
            lr = self.control_lr

        W1_t_control, W2_t_control = self.get_weights(self.time_span)
        L_t = self.get_loss_function(W1=W1_t_control, W2=W2_t_control)
        C_t = self.control_cost()

        instant_reward_rate = self.gamma**(self.time_span)*(-self.reward_convertion*L_t-C_t)
        cumulated_R = torch.sum(instant_reward_rate)*self.dt

        cumulated_R.backward()

        # for iters in range(inner_loop):
        self.update_layer = 0
        self.g1, self.g1_tilda = self._update_g(self.g1, lr=lr)
        self.update_layer = 1
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