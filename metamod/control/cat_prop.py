import torch
from metamod.control import LinearNetEq
import numpy as np


class CatPropControl(LinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0, engagement_coef=None,
                 control_lower_bound=0.0, control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4, type_of_engage="vector"):

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
        self.type_of_engage = type_of_engage
        self.cost_offset = 0.0 if type_of_engage == "active" else 1.0

        self.input_dim = self.W1.shape[1]
        self.hidden_dim = self.W1.shape[0]
        self.output_dim = self.W2.shape[0]

        if engagement_coef is None:
            engagement_coef = np.ones((n_steps, self.output_dim))

        self.engagement_coef = torch.tensor(engagement_coef, requires_grad=True, dtype=self.dtype,
                                            device=self.device)

        time_span = np.arange(0, n_steps) * learning_rate
        self.dt = time_span[1] - time_span[0]
        self.time_span = torch.from_numpy(time_span).requires_grad_(False).type(self.dtype).to(self.device)

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
        cumulated_R = torch.sum(instant_reward_rate)*self.dt

        cumulated_R.backward()
        self.engagement_coef = self._update_phis(eng_coef=self.engagement_coef, lr=lr)
        self._bound_control()

        if get_numpy:
            return cumulated_R.detach().cpu().numpy()
        else:
            return cumulated_R

    def weight_der(self, t, W1, W2, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        diagonal_coef = torch.diag(self.engagement_coef[t_index, :]**2)

        dW1 = W2.T @ diagonal_coef @ self.in_out_cov.T - W2.T @ diagonal_coef @ W2 @ W1 @ self.in_cov - self.reg_coef * W1
        dW2 = diagonal_coef @ self.in_out_cov.T @ W1.T - diagonal_coef @ W2 @ W1 @ self.in_cov @ W1.T - self.reg_coef * W2

        dW2 /= self.time_constant
        dW1 /= self.time_constant

        return dW1, dW2

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


if __name__ == "__main__":
    from metamod.tasks import SemanticTask
    from metamod.control import LinearNetEq
    from metamod.networks import LinearNet
    from metamod.trainers import two_layer_engage_training, two_layer_training

    run_name = "cat_prop"
    results_path = "../results"
    results_dict = {}

    dataset_params = {"batch_size": 32,
                      "h_levels": 3}

    dataset = SemanticTask(**dataset_params)

    model_params = {"learning_rate": 5e-3,
                    "hidden_dim": 10,
                    "intrinsic_noise": 0.0,
                    "reg_coef": 0.0,
                    "input_dim": dataset.input_dim,
                    "output_dim": dataset.output_dim,
                    "W1_0": None,
                    "W2_0": None}

    model_params["input_dim"] = dataset.input_dim
    model_params["output_dim"] = dataset.output_dim

    model_params["W1_0"] = np.random.normal(scale=1e-4,
                                            size=(model_params["hidden_dim"], model_params["input_dim"]))
    model_params["W2_0"] = np.random.normal(scale=1e-4,
                                            size=(model_params["output_dim"], model_params["hidden_dim"]))

    model = LinearNet(**model_params)

    n_steps = 50
    save_weights_every = 20

    iters, loss, weights_iter, weights = two_layer_training(model=model, dataset=dataset, n_steps=n_steps,
                                                            save_weights_every=save_weights_every)

    results_dict["iters"] = iters
    results_dict["Loss_t_sim"] = loss
    results_dict["weights_sim"] = weights
    results_dict["weights_iters_sim"] = weights_iter

    init_W1 = weights[0][0, ...]
    init_W2 = weights[1][0, ...]

    init_weights = [init_W1, init_W2]
    input_corr, output_corr, input_output_corr, expected_y, expected_x = dataset.get_correlation_matrix()

    time_span = np.arange(0, len(iters)) * model_params["learning_rate"]
    results_dict["time_span"] = time_span

    equation_params = {"in_cov": input_corr,
                       "out_cov": output_corr,
                       "in_out_cov": input_output_corr,
                       # "expected_y": expected_y,
                       # "expected_x": expected_x,
                       "init_weights": init_weights,
                       "n_steps": n_steps,
                       "reg_coef": model_params["reg_coef"],
                       "intrinsic_noise": model_params["intrinsic_noise"],
                       "learning_rate": model_params["learning_rate"],
                       "time_constant": 1.0}

    solver = LinearNetEq(**equation_params)

    control_params = {**equation_params,
                      "control_lower_bound": 0.0,
                      "control_upper_bound": 2.0,
                      "gamma": 0.99,
                      "cost_coef": 0.1,
                      "reward_convertion": 1.0,
                      "init_g": None,
                      "control_lr": 1.0}

    control = CatPropControl(**control_params)

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
    results_dict["engagement_coef"] = control.engagement_coef

    iter_control = 10
    control_params["iters_control"] = iter_control
    cumulated_reward = []

    for i in range(iter_control):
        R = control.train_step(get_numpy=True)
        print("cumulated reward:", R)
        cumulated_reward.append(R)
    cumulated_reward = np.array(cumulated_reward).astype(float)
    results_dict["cumulated_reward_opt"] = cumulated_reward