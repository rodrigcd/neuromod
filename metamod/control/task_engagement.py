import torch
from metamod.control import LinearNetEq
import numpy as np


class LinearNetTaskEngEq(LinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 task_output_index=(), task_input_index=(), engagement_coef=None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        if len(task_output_index) != engagement_coef.shape[1]:
            raise Exception("task index not equal to number of eng coef")

        # List of expected values, as well as every cov matrix in the input
        self.tasks_expected_x = expected_x
        self.tasks_expected_y = expected_y
        self.task_output_index = task_output_index
        self.task_input_index = task_input_index
        self.engagement_coef = torch.tensor(engagement_coef, requires_grad=True, dtype=self.dtype, device=self.device)
        self.n_tasks = len(task_output_index)
        self.tasks_in_cov = in_cov
        self.tasks_in_out_cov = in_out_cov
        self.tasks_out_cov = out_cov

        self._get_overall_covariance()

        super().__init__(self.overall_in_out_cov, self.overall_in_cov, self.overall_out_cov,
                         init_weights, reg_coef, intrinsic_noise, learning_rate, n_steps, time_constant)

    def _get_overall_covariance(self):
        self.input_dim = np.sum([len(x) for x in self.tasks_expected_x])
        self.output_dim = np.sum([len(y) for y in self.tasks_expected_y])

        # Big covariance matrix relating tasks
        self.overall_in_cov = np.zeros(shape=(self.input_dim, self.input_dim))
        self.overall_in_out_cov = np.zeros(shape=(self.input_dim, self.output_dim))
        self.overall_out_cov = np.zeros(shape=(self.output_dim, self.output_dim))
        self.in_out_cov_pertask = []

        for j, expected_y in enumerate(self.tasks_expected_y):
            per_task_in_out = np.zeros(shape=(self.input_dim, self.output_dim))
            for i, expected_x in enumerate(self.tasks_expected_x):
                if i == j:
                    in_cov_fill = self.tasks_in_cov[i]
                    in_out_cov_fill = self.tasks_in_out_cov[i]
                    out_cov_fill = self.tasks_out_cov[i]
                else:
                    in_cov_fill = self.tasks_expected_x[i][:, np.newaxis] @ self.tasks_expected_x[j][np.newaxis, :]
                    in_out_cov_fill = self.tasks_expected_x[i][:, np.newaxis] @ self.tasks_expected_y[j][np.newaxis, :]
                    out_cov_fill = self.tasks_expected_y[i][:, np.newaxis] @ self.tasks_expected_y[j][np.newaxis, :]
                in_index_i = self.task_input_index[i]
                in_index_j = self.task_input_index[j]
                out_index_i = self.task_output_index[i]
                out_index_j = self.task_output_index[j]

                self.overall_in_cov[in_index_i[0]:in_index_i[1], in_index_j[0]:in_index_j[1]] = in_cov_fill
                self.overall_in_out_cov[in_index_i[0]:in_index_i[1], out_index_j[0]:out_index_j[1]] = in_out_cov_fill
                self.overall_out_cov[out_index_i[0]:out_index_i[1], out_index_j[0]:out_index_j[1]] = out_cov_fill
                per_task_in_out[in_index_i[0]:in_index_i[1], out_index_j[0]:out_index_j[1]] = in_out_cov_fill
            self.in_out_cov_pertask.append(torch.from_numpy(per_task_in_out).requires_grad_(False).type(self.dtype).to(self.device))

    def weight_der(self, t, W1, W2, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        dW2 = -self.reg_coef*W2
        dW1 = -self.reg_coef*W1
        for i in range(self.n_tasks):
            task_index = self.task_output_index[i]
            sigma_xyi = self.in_out_cov_pertask[i]
            padded_W2 = self.get_padded_weight(W2, task_index)

            dW2 += self.engagement_coef[t_index, i]*((sigma_xyi.T - padded_W2 @ W1 @ self.in_cov) @ W1.T)
            dW1 += self.engagement_coef[t_index, i]*(padded_W2.T @ (sigma_xyi.T - padded_W2 @ W1 @ self.in_cov))

        dW2 /= self.time_constant
        dW1 /= self.time_constant

        return dW1, dW2

    def get_padded_weight(self, W, task_index):
        if len(W.shape) == 2:
            to_pad_W = W[task_index[0]:task_index[1], :]
            padding_top, padding_bottom = task_index[0], W.shape[0] - task_index[1]
            padding = (0, 0, padding_top, padding_bottom)
            padding_function = torch.nn.ConstantPad2d(padding=padding, value=0)
        else:
            to_pad_W = W[:, task_index[0]:task_index[1], :]
            padding_top, padding_bottom = task_index[0], W.shape[1] - task_index[1]
            padding = (0, 0, padding_top, padding_bottom, 0, 0)
            padding_function = torch.nn.ConstantPad3d(padding=padding, value=0)

        # variables per tasks
        padded_W = padding_function(to_pad_W)
        return padded_W

    # def get_loss_function(self, W1, W2, get_numpy=False):
    #     # Here the weights have shape (t, weight_shape[0], weight_shape[1])
    #     if isinstance(W1, np.ndarray):
    #         W2 = torch.from_numpy(W2).type(self.dtype).to(self.device)
    #         W1 = torch.from_numpy(W1).type(self.dtype).to(self.device)
    #
    #     tasks_loss = 0
    #     for i in range(self.n_tasks):
    #         task_index = self.task_output_index[i]
    #         padded_W2 = self.get_padded_weight(W2, task_index)
    #
    #         W_t = padded_W2 @ W1
    #         task_out_cov_i = torch.from_numpy(self.tasks_out_cov[i]).requires_grad_(False).type(self.dtype).to(self.device)
    #         sigma_xyi = self.in_out_cov_pertask[i]
    #
    #         tasks_loss += self.engagement_coef[:, i]*0.5*(torch.trace(task_out_cov_i)
    #                            - torch.diagonal(2*sigma_xyi @ W_t, dim1=-2, dim2=-1).sum(-1)
    #                            + torch.diagonal(self.in_cov @ torch.transpose(W_t, dim0=-1, dim1=-2) @ W_t,
    #                                             dim1=-2, dim2=-1).sum(-1)) + 0.5*W_t.shape[1]*self.intrinsic_noise**2
    #
    #     L2 = (self.reg_coef/2.0)*(torch.sum(W1**2, (-1, -2)) + torch.sum(W2**2, (-1, -2)))
    #     L = tasks_loss + L2
    #     if get_numpy:
    #         return L.detach().cpu().numpy()
    #     else:
    #         return L


class LinearNetTaskEngControl(LinearNetTaskEngEq):

    def __init__(self, in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 task_output_index=(), task_input_index=(), engagement_coef=None,
                 control_lower_bound=0.0, control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4, type_of_engage="active"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        super().__init__(in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant, task_output_index,
                         task_input_index, engagement_coef)

        self.control_upper_bound = control_upper_bound
        self.control_lower_bound = control_lower_bound
        self.gamma = gamma
        self.cost_coef = cost_coef
        self.reward_convertion = reward_convertion
        self.control_lr = control_lr
        self.type_of_engage = type_of_engage
        self.cost_offset = 0.0 if type_of_engage == "active" else 1.0

        self.input_dim = self.W1.shape[1]
        self.hidden_dim = self.W1.shape[0]
        self.output_dim = self.W2.shape[0]

        time_span = np.arange(0, n_steps) * learning_rate
        self.dt = time_span[1] - time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)
        # self._bound_control()

    def _bound_control(self):
        if self.control_upper_bound is None and self.control_lower_bound is not None:
            self.engagement_coef.data.clamp_(min=self.control_lower_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is not None:
            self.engagement_coef.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is None:
            self.engagement_coef.data.clamp_(max=self.control_upper_bound)

    def control_cost(self, get_numpy=False):
        cost = self.cost_coef*torch.sum((self.engagement_coef - self.cost_offset) ** 2, dim=-1)
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

    def train_step(self, get_numpy=False, lr=None):
        if lr is None:
            lr = self.control_lr

        W1_t_control, W2_t_control = self.get_weights(self.time_span)
        L_t = self.get_loss_function(W1=W1_t_control, W2=W2_t_control)
        C_t = self.control_cost()

        instant_reward_rate = self.gamma**(self.time_span)*(-self.reward_convertion*L_t-C_t)
        cumulated_R = torch.sum(instant_reward_rate)*self.dt

        cumulated_R.backward()
        self.engagement_coef = self._update_phis(eng_coef=self.engagement_coef, lr=lr)
        self._bound_control()

        if get_numpy:
            return cumulated_R.detach().cpu().numpy()
        else:
            return cumulated_R

    def _update_phis(self, eng_coef, lr):
        with torch.no_grad():
            eng_coef += lr*eng_coef.grad
            eng_coef.grad = None
            if self.control_upper_bound is None and self.control_lower_bound is not None:
                eng_coef.data.clamp_(min=self.control_lower_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is not None:
                eng_coef.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
            elif self.control_upper_bound is not None and self.control_lower_bound is None:
                eng_coef.data.clamp_(max=self.control_upper_bound)
        return eng_coef
