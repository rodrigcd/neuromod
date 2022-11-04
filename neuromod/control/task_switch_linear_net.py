import torch
import numpy as np
from neuromod.control import LinearNetEq, LinearNetControl


class TaskSwitchLinearNetEq(LinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 change_task_every=200):

        super().__init__(in_out_cov[0], in_cov[0], out_cov[0], init_weights, reg_coef, intrinsic_noise,
                         learning_rate, n_steps, time_constant)

        self.in_out_cov_list = in_out_cov
        self.in_cov_list = in_cov
        self.out_cov_list = out_cov
        self.change_task_every = change_task_every
        self.current_dataset_id = 0
        self._generate_cov_matrices()

    def _generate_cov_matrices(self):
        self.broad_out_cov = []
        self.broad_in_cov = []
        self.broad_in_out_cov = []
        for t in range(self.n_steps):
            self.update_covariances(t_index=t)
            self.broad_out_cov.append(self.out_cov)
            self.broad_in_cov.append(self.in_cov)
            self.broad_in_out_cov.append(self.in_out_cov)
        self.broad_in_out_cov = torch.stack(self.broad_in_out_cov, dim=0)
        self.broad_out_cov = torch.stack(self.broad_out_cov, dim=0)
        self.broad_in_cov = torch.stack(self.broad_in_cov, dim=0)

    def update_covariances(self, t_index):
        section = np.floor(t_index/self.change_task_every)
        self.current_dataset_id = int(section % len(self.in_out_cov_list))

        self.in_out_cov = torch.from_numpy(self.in_out_cov_list[self.current_dataset_id]).requires_grad_(
            False).type(self.dtype).to(self.device)
        self.in_cov = torch.from_numpy(self.in_cov_list[self.current_dataset_id]).requires_grad_(
            False).type(self.dtype).to(self.device)
        self.out_cov = torch.from_numpy(self.out_cov_list[self.current_dataset_id]).requires_grad_(
            False).type(self.dtype).to(self.device)

    def weight_der(self, t, W1, W2, t_index=None):
        self.update_covariances(t_index)
        dW1, dW2 = super().weight_der(t, W1, W2, t_index)
        return dW1, dW2

    def get_loss_function(self, W1, W2, get_numpy=False):
        W_t = W2 @ W1
        if isinstance(W_t, np.ndarray):
            W_t = torch.from_numpy(W_t).type(self.dtype).to(self.device)
            W2 = torch.from_numpy(W2).type(self.dtype).to(self.device)
            W1 = torch.from_numpy(W1).type(self.dtype).to(self.device)
        L1 = 0.5 * (torch.diagonal(self.broad_out_cov, dim1=-2, dim2=-1).sum(-1)
                    - torch.diagonal(2 * self.broad_in_out_cov @ W_t, dim1=-2,
                                     dim2=-1).sum(-1)
                    + torch.diagonal(self.broad_in_cov @ torch.transpose(W_t, dim0=-1, dim1=-2) @ W_t,
                                     dim1=-2, dim2=-1).sum(-1)) + 0.5 * W_t.shape[1] * self.intrinsic_noise ** 2
        L2 = (self.reg_coef / 2.0) * (torch.sum(W1 ** 2, (-1, -2)) + torch.sum(W2 ** 2, (-1, -2)))
        L = L1 + L2
        if get_numpy:
            return L.detach().cpu().numpy()
        else:
            return L


class TaskSwitchLinearNetControl(TaskSwitchLinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef, intrinsic_noise, learning_rate=1e-5,
                 n_steps=10000, time_constant=1.0, change_task_every=200, control_lower_bound=0.0,
                 control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3, reward_convertion=1.0,
                 control_lr=1e-4):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        super().__init__(in_out_cov, in_cov, out_cov, init_weights, reg_coef, intrinsic_noise, learning_rate, n_steps,
                         time_constant, change_task_every)

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
        self.dt = time_span[1] - time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)
        self.g1, self.g1_tilda = self._get_g(init_g, shape=(n_steps, self.W1.shape[0], self.W1.shape[1]))
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
        self.update_covariances(t_index)

        g1_tilda = self.g1_tilda[t_index, :, :]
        g2_tilda = self.g2_tilda[t_index, :, :]

        W1_tilda = W1 * g1_tilda
        W2_tilda = W2 * g2_tilda

        dW1 = ((W2_tilda.T @ (self.in_out_cov.T - W2_tilda @ W1_tilda @ self.in_cov)) * g1_tilda - self.reg_coef * W1) / self.time_constant
        dW2 = (((self.in_out_cov.T - W2_tilda @ W1_tilda @ self.in_cov) @ W1_tilda.T) * g2_tilda - self.reg_coef * W2) / self.time_constant
        return dW1, dW2

    def get_loss_function(self, W1, W2, get_numpy=False):
        if isinstance(W1, np.ndarray):
            W2 = torch.from_numpy(W2).type(self.dtype).to(self.device)
            W1 = torch.from_numpy(W1).type(self.dtype).to(self.device)

        W_t = (W2 * self.g2_tilda) @ (W1 * self.g1_tilda)
        L1 = 0.5 * (torch.diagonal(self.broad_out_cov, dim1=-2, dim2=-1).sum(-1)
                    - torch.diagonal(2 * self.broad_in_out_cov @ W_t, dim1=-2,
                                     dim2=-1).sum(-1)
                    + torch.diagonal(self.broad_in_cov @ torch.transpose(W_t, dim0=-1, dim1=-2) @ W_t,
                                     dim1=-2, dim2=-1).sum(-1)) + 0.5 * W_t.shape[1] * self.intrinsic_noise ** 2
        L2 = (self.reg_coef / 2.0) * (torch.sum(W1 ** 2, (-1, -2)) + torch.sum(W2 ** 2, (-1, -2)))
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

        instant_reward_rate = self.gamma ** (self.time_span) * (-self.reward_convertion * L_t - C_t)
        cumulated_R = torch.sum(instant_reward_rate) * self.dt

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
            g += lr * g.grad
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


if __name__ == "__main__":
    import numpy as np
    from neuromod.tasks import TaskSwitch, AffineCorrelatedGaussian
    from neuromod.trainers import two_layer_training
    from neuromod.networks import LinearNet
    from neuromod.control import TaskSwitchLinearNetEq

    run_name = "task_switch_test"
    results_path = "../results"

    n_steps = 21000
    save_weights_every = 20
    iter_control = 10

    results_dict = {}

    # Init dataset
    batch_size = 2048
    dataset1_params = {"mu_vec": (3.0, 1.0), "sigma_vec": (1.0, 1.0), "dependence_parameter": 0.8,
                       "batch_size": batch_size}
    dataset2_params = {"mu_vec": (-2.0, 2.0), "sigma_vec": (1.0, 1.0), "dependence_parameter": 0.2,
                       "batch_size": batch_size}
    dataset_params = {"dataset1_params": dataset1_params,
                      "dataset2_params": dataset2_params,
                      "change_tasks_every": 1500}

    model_params = {"learning_rate": 5e-3,
                    "hidden_dim": 6,
                    "intrinsic_noise": 0.05,
                    "reg_coef": 0.0,
                    "W1_0": None,
                    "W2_0": None}

    control_params = {"control_lower_bound": -0.5,
                      "control_upper_bound": 0.5,
                      "gamma": 0.99,
                      "cost_coef": 0.3,
                      "reward_convertion": 1.0,
                      "init_g": None,
                      "control_lr": 10.0}

    dataset = TaskSwitch(dataset_classes=(AffineCorrelatedGaussian, AffineCorrelatedGaussian),
                         dataset_list_params=(dataset1_params, dataset2_params),
                         change_tasks_every=dataset_params["change_tasks_every"])

    model_params["input_dim"] = dataset.input_dim
    model_params["output_dim"] = dataset.output_dim

    model = LinearNet(**model_params)

    iters, loss, weights_iter, weights = two_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                            save_weights_every=save_weights_every)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter

    # Solving equation
    init_W1 = weights[0][0, ...]
    init_W2 = weights[1][0, ...]

    init_weights = [init_W1, init_W2]
    input_corr, output_corr, input_output_corr, expected_y, expected_x = dataset.get_correlation_matrix()

    time_span = np.arange(0, len(iters)) * model_params["learning_rate"]
    results_dict["time_span"] = time_span

    equation_params = {"in_cov": input_corr,
                       "out_cov": output_corr,
                       "in_out_cov": input_output_corr,
                       "init_weights": init_weights,
                       "n_steps": n_steps,
                       "reg_coef": model_params["reg_coef"],
                       "intrinsic_noise": model_params["intrinsic_noise"],
                       "learning_rate": model_params["learning_rate"],
                       "change_task_every": dataset_params["change_tasks_every"],
                       "time_constant": 1.0}

    solver = TaskSwitchLinearNetEq(**equation_params)

    control_params = {**control_params, **equation_params}
    control = TaskSwitchLinearNetControl(**control_params)

    W1_t, W2_t = solver.get_weights(time_span, get_numpy=True)
    Loss_t = solver.get_loss_function(W1_t, W2_t, get_numpy=True)

    results_dict["W1_t_eq"] = W1_t
    results_dict["W2_t_eq"] = W2_t
    results_dict["Loss_t_eq"] = Loss_t

    W1_t_control, W2_t_control = control.get_weights(time_span, get_numpy=True)
    Loss_t_control = control.get_loss_function(W1_t_control, W2_t_control, get_numpy=True)

    results_dict["W1_t_control_init"] = W1_t_control
    results_dict["W2_t_control_init"] = W2_t_control
    results_dict["Loss_t_control_init"] = Loss_t_control
    results_dict["control_signal_init"] = (control.g1_tilda, control.g2_tilda)

    control_params["iters_control"] = iter_control
    cumulated_reward = []

    for i in range(iter_control):
        R = control.train_step(get_numpy=True, lr=10.0)
        print(R)
        cumulated_reward.append(R)
    cumulated_reward = np.array(cumulated_reward).astype(float)
    results_dict["cumulated_reward_opt"] = cumulated_reward

    W1_t_opt, W2_t_opt = control.get_weights(time_span, get_numpy=True)
    Loss_t_opt = control.get_loss_function(W1_t_opt, W2_t_opt, get_numpy=True)

    results_dict["W1_t_control_opt"] = W1_t_opt
    results_dict["W2_t_control_opt"] = W2_t_opt
    results_dict["Loss_t_control_opt"] = Loss_t_opt