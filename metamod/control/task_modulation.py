import torch
from metamod.control import LinearNetEq
import numpy as np


class LinearNetTaskModEq(LinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0, engagement_coef=None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        # List of expected values, as well as every cov matrix in the input
        self.tasks_expected_x = expected_x
        self.tasks_expected_y = expected_y

        self.n_tasks = len(in_out_cov)
        self.tasks_in_cov = in_cov
        self.tasks_in_out_cov = in_out_cov
        self.tasks_out_cov = out_cov

        self._get_overall_covariance()
        self.softmax_torch = torch.nn.Softmax(dim=0)

        if engagement_coef is None:
            engagement_coef = np.ones((n_steps, self.n_tasks))

        self.engagement_coef = torch.tensor(engagement_coef, requires_grad=True, dtype=self.dtype,
                                            device=self.device)

        super().__init__(in_out_cov[0], in_cov[0], out_cov[0],
                         init_weights, reg_coef, intrinsic_noise, learning_rate, n_steps, time_constant)

    def _get_overall_covariance(self):
        self.input_dim = len(self.tasks_expected_x[0])
        self.output_dim = len(self.tasks_expected_y[0])
        self.in_cov_array = []
        self.in_out_cov_array = []
        self.out_cov_array = []

        for j, expected_y in enumerate(self.tasks_expected_y):
            in_cov_row = []
            out_cov_row = []
            in_out_cov_row = []

            for i, expected_x in enumerate(self.tasks_expected_x):
                if i == j:
                    in_cov_fill = self.tasks_in_cov[i]
                    in_out_cov_fill = self.tasks_in_out_cov[i]
                    out_cov_fill = self.tasks_out_cov[i]
                else:
                    in_cov_fill = self.tasks_expected_x[i][:, np.newaxis] @ self.tasks_expected_x[j][np.newaxis, :]
                    in_out_cov_fill = self.tasks_expected_x[i][:, np.newaxis] @ self.tasks_expected_y[j][np.newaxis, :]
                    out_cov_fill = self.tasks_expected_y[i][:, np.newaxis] @ self.tasks_expected_y[j][np.newaxis, :]
                in_cov_row.append(in_cov_fill)
                out_cov_row.append(out_cov_fill)
                in_out_cov_row.append(in_out_cov_fill)

            self.in_cov_array.append(np.stack(in_cov_row, axis=0)[:, np.newaxis, :, :])
            self.in_out_cov_array.append(np.stack(in_out_cov_row, axis=0)[:, np.newaxis, :, :])
            self.out_cov_array.append(np.stack(out_cov_row, axis=0)[:, np.newaxis, :, :])

        self.in_cov_array = np.concatenate(self.in_cov_array, axis=1)
        self.in_out_cov_array = np.concatenate(self.in_out_cov_array, axis=1)
        self.out_cov_array = np.concatenate(self.out_cov_array, axis=1)

        self.in_cov_array = torch.tensor(self.in_cov_array, requires_grad=False,
                                         dtype=self.dtype, device=self.device)
        self.in_out_cov_array = torch.tensor(self.in_out_cov_array, requires_grad=False,
                                             dtype=self.dtype, device=self.device)
        self.out_cov_array = torch.tensor(self.out_cov_array, requires_grad=False,
                                          dtype=self.dtype, device=self.device)

    def compute_covariance(self, sigmoid_coef):
        simetric_S = sigmoid_coef[:, None] @ sigmoid_coef[None, :]
        Sigma_x = torch.sum(self.in_cov_array * simetric_S[:, :, None, None], dim=(0, 1))

        Sigma_xy_tau = torch.stack([torch.sum(self.in_out_cov_array[:, i, :, :]*sigmoid_coef[:, None, None], dim=0) for i in range(self.n_tasks)],
                                    dim=0)
        return Sigma_x, Sigma_xy_tau

    def weight_der(self, t, W1, W2, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        eng_coef = self.engagement_coef[t_index, :]
        softmax_coef = self.softmax_torch(eng_coef)
        dW2 = -self.reg_coef*W2
        dW1 = -self.reg_coef*W1
        Sigma_x, Sigma_xy_tau = self.compute_covariance(softmax_coef)

        for i in range(self.n_tasks):
            dW2 += softmax_coef[i]*((Sigma_xy_tau[i, :, :].T - W2 @ W1 @ Sigma_x) @ W1.T)
            dW1 += softmax_coef[i]*(W2.T @ (Sigma_xy_tau[i, :, :].T - W2 @ W1 @ Sigma_x))

        dW2 /= self.time_constant
        dW1 /= self.time_constant

        return dW1, dW2


class LinearNetTaskModControl(LinearNetTaskModEq):

    def __init__(self, in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0, engagement_coef=None,
                 control_lower_bound=0.0, control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4, type_of_engage="vector"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        super().__init__(in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant, engagement_coef)

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
        self.optimizer = torch.optim.Adam([self.engagement_coef, ], lr=self.control_lr)
        # self._bound_control()

    def _bound_control(self):
        if self.control_upper_bound is None and self.control_lower_bound is not None:
            self.engagement_coef.data.clamp_(min=self.control_lower_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is not None:
            self.engagement_coef.data.clamp_(min=self.control_lower_bound, max=self.control_upper_bound)
        elif self.control_upper_bound is not None and self.control_lower_bound is None:
            self.engagement_coef.data.clamp_(max=self.control_upper_bound)

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
        cumulated_R = -torch.sum(instant_reward_rate)*self.dt

        self.optimizer.zero_grad()
        cumulated_R.backward()
        self.optimizer.step()
        # cumulated_R.backward()
        # self.engagement_coef = self._update_phis(eng_coef=self.engagement_coef, lr=lr)
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
