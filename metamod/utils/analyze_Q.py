import numpy as np
from metamod.utils import ResultsManager
import torch
from tqdm import tqdm


class QAnalysis(object):

    def __init__(self, results_path, optimizing_window=100, verbose=False, q_iters=2000, q_opt_lr=0.01):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.verbose = verbose
        self.results_path = results_path
        self.optimizing_window = optimizing_window
        self.Q_iters = q_iters
        self.Q_opt_lr = q_opt_lr
        self.results = ResultsManager(results_path, verbose=verbose)
        self.net_eq = self.results.params["equation_params"]["solver"]
        self.reg_coef = self.net_eq.reg_coef
        self.time_constant = self.net_eq.time_constant
        self.time_span = np.arange(self.optimizing_window) * self.net_eq.learning_rate
        self.dt = self.time_span[1] - self.time_span[0]
        self.intrinsic_noise = self.net_eq.intrinsic_noise
        self.dataset = self.results.params["dataset_params"]["dataset"]
        self.change_tasks_every = self.dataset.change_tasks_every
        self.n_steps = self.results.params["equation_params"]["n_steps"]
        self.iters = np.arange(self.n_steps)
        self._compute_regression_solution()

    def _compute_fixed_point(self):
        fixed_point_W1 = []
        fixed_point_W2 = []
        G1_tilda, G2_tilda = self.results.results["control_signal"]
        G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
        W1_tilde = G1_tilda * self.control_W1
        W2_tilde = G2_tilda * self.control_W2
        task2_solution = self.task2_solution
        task1_solution = self.task1_solution
        task_solutions = [task1_solution, task2_solution]
        baseline_fixed_point = []
        control_fixed_point = []
        for t_index in self.iters:
            section = np.floor(t_index / self.change_tasks_every)
            current_dataset_id = int(section % len(self.dataset.dataset_classes))
            baseline_fixed_point.append(task_solutions[current_dataset_id])

            G1 = G1_tilda[t_index, :, :]
            G2 = G1_tilda[t_index, :, :]
            W1_tilde_t = W1_tilde[t_index, :, :]
            W2_tilde_t = W2_tilde[t_index, :, :]

            W1_fixed_pont = (baseline_fixed_point[-1] @ np.linalg.inv(W1_tilde_t @ W1_tilde_t.T) @ W1_tilde_t.T)/G1
            W2_fixed_point = (W2_tilde_t.T @ np.linalg.inv(W2_tilde_t @ W2_tilde_t.t) @ baseline_fixed_point[-1])/G2

            control_fixed_point.append(W2_fixed_point @ W1_fixed_pont)

        return np.stack(baseline_fixed_point), np.stack(control_fixed_point)


    def _compute_regression_solution(self):
        if self.verbose:
            print("Computing regression solution")
        self.task1_solution = self.dataset.datasets[0].get_linear_regression_solution()
        self.task2_solution = self.dataset.datasets[1].get_linear_regression_solution()
        self.cov_matrix_task1 = self.dataset.datasets[0].get_correlation_matrix()
        self.cov_matrix_task2 = self.dataset.datasets[1].get_correlation_matrix()

        self.best_loss_task1 = self.get_optimal_loss(self.task1_solution, self.cov_matrix_task1[1],
                                                     self.cov_matrix_task1[2], self.cov_matrix_task1[0])
        self.best_loss_task2 = self.get_optimal_loss(self.task2_solution, self.cov_matrix_task2[1],
                                                     self.cov_matrix_task2[2], self.cov_matrix_task2[0])

        u, s, vh = np.linalg.svd(self.task1_solution, full_matrices=True)
        s = np.concatenate([np.diag(s), np.zeros((len(s), 1))], axis=1)
        s_prime = np.concatenate([s, np.zeros((1, s.shape[1]))], axis=0)
        Us_1 = np.linalg.pinv(u @ np.sqrt(s))
        Vs_1 = np.linalg.pinv(np.sqrt(s_prime) @ vh)
        self.svd_task1 = {"U": u, "S": s, "S_prime": s_prime, "V_T": vh,
                          "Us_1": Us_1, "Vs_1": Vs_1}

        u, s, vh = np.linalg.svd(self.task2_solution, full_matrices=True)
        s = np.concatenate([np.diag(s), np.zeros((len(s), 1))], axis=1)
        s_prime = np.concatenate([s, np.zeros((1, s.shape[1]))], axis=0)
        Us_1 = np.linalg.pinv(u @ np.sqrt(s))
        Vs_1 = np.linalg.pinv(np.sqrt(s_prime) @ vh)
        self.svd_task2 = {"U": u, "S": s, "S_prime": s_prime, "V_T": vh,
                          "Us_1": Us_1, "Vs_1": Vs_1}

        self.baseline_W1 = self.results.results["W1_t_eq"]
        self.baseline_W2 = self.results.results["W2_t_eq"]
        self.control_W1 = self.results.results["W1_t_control_opt"]
        self.control_W2 = self.results.results["W2_t_control_opt"]

        self.baseline_Q1, self.baseline_Q2, self.task_id_t = self._estimate_q_from_weights(self.baseline_W1,
                                                                                           self.baseline_W2)
        self.control_Q1, self.control_Q2, _ = self._estimate_q_from_weights(self.control_W1,
                                                                            self.control_W2)

        G1_tilda, G2_tilda = self.results.results["control_signal"]
        G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
        W1_tilde = G1_tilda * self.control_W1
        W2_tilde = G2_tilda * self.control_W2
        self.tilde_Q1, self.tilde_Q2, _ = self._estimate_q_from_weights(W1_tilde, W2_tilde)

    def estimate_best_q(self, task1_svd, task2_cov):
        # Move variables to device
        gpu_svd = {}
        for key, value in task1_svd.items():
            gpu_svd[key] = torch.tensor(value, dtype=self.dtype, device=self.device, requires_grad=False)
        gpu_cov = []
        for value in task2_cov:
            gpu_cov.append(torch.tensor(value, dtype=self.dtype, device=self.device, requires_grad=False))

        # Initialize Q
        hidden_units = self.net_eq.hidden_dim
        Q_matrix = np.random.normal(loc=0, scale=2.0, size=(hidden_units, task1_svd["S"].shape[1]))
        Q_matrix[:, -1] = 0
        Q_matrix = torch.tensor(Q_matrix, requires_grad=True, device=self.device, dtype=self.dtype)
        Q_inverse = torch.linalg.pinv(Q_matrix)

        # Build weights
        W2_task_init = gpu_svd["U"] @ torch.sqrt(gpu_svd["S"]) @ Q_inverse
        W1_task_init = Q_matrix @ torch.sqrt(gpu_svd["S_prime"]).T @ gpu_svd["V_T"]

        # Replace init weights and optimizing period
        W1 = W1_task_init
        W2 = W2_task_init

        input_corr, output_corr, input_output_corr, expected_y, expected_x = gpu_cov

        # Run loop
        loss_trajectories = []
        for i in tqdm(range(self.Q_iters), disable=(not self.verbose)):
            W1_t, W2_t = self.get_weights(self.time_span, W1, W2,
                                          in_out_cov=input_output_corr,
                                          in_cov=input_corr)
            L_t = self.get_loss_function(W1=W1_t, W2=W2_t,
                                         in_out_cov=input_output_corr,
                                         in_cov=input_corr,
                                         out_cov=output_corr)
            loss_trajectories.append(L_t.detach().cpu().numpy())
            loss_integral = torch.sum(L_t)*self.dt
            # print("Loss integral:", loss_integral.detach().cpu().numpy())
            # print("Init loss:", L_t.detach().cpu().numpy()[0])
            loss_integral.backward()
            with torch.no_grad():
                Q_matrix -= self.Q_opt_lr * Q_matrix.grad
            Q_inverse = torch.linalg.pinv(Q_matrix)
            # Build weights
            W2_task_init = gpu_svd["U"] @ torch.sqrt(gpu_svd["S"]) @ Q_inverse
            W1_task_init = Q_matrix @ torch.sqrt(gpu_svd["S_prime"]).T @ gpu_svd["V_T"]
            W1 = W1_task_init
            W2 = W2_task_init
        return loss_trajectories, Q_matrix.detach().cpu().numpy()

    def weight_der(self, t, W1, W2, in_out_cov, in_cov, t_index=None):
        if t_index is None:
            t_index = (self.time_span == t).nonzero(as_tuple=True)[0][0]

        dW1 = (W2.T @ (in_out_cov.T - W2 @ W1 @ in_cov) - self.reg_coef * W1)/self.time_constant
        dW2 = ((in_out_cov.T - W2 @ W1 @ in_cov) @ W1.T - self.reg_coef * W2)/self.time_constant
        return dW1, dW2

    def get_weights(self, time_span, W1_0, W2_0, in_out_cov, in_cov, get_numpy=False):
        W1_t = []
        W2_t = []
        current_W1 = torch.clone(W1_0)
        current_W2 = torch.clone(W2_0)
        W1_t.append(current_W1)
        W2_t.append(current_W2)
        for i, t in enumerate(time_span[:-1]):
            dW1, dW2 = self.weight_der(t, current_W1, current_W2, in_out_cov, in_cov, t_index=i)
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

    def get_loss_function(self, W1, W2, in_out_cov, in_cov, out_cov, get_numpy=False):
        W_t = W2 @ W1
        L1 = 0.5*(torch.trace(out_cov) - torch.diagonal(2*in_out_cov @ W_t, dim1=-2, dim2=-1).sum(-1)
                          + torch.diagonal(in_cov @ torch.transpose(W_t, dim0=-1, dim1=-2) @ W_t,
                                           dim1=-2, dim2=-1).sum(-1)) + 0.5*W_t.shape[1]*self.intrinsic_noise**2
        L2 = (self.reg_coef/2.0)*(torch.sum(W1**2, (-1, -2)) + torch.sum(W2**2, (-1, -2)))
        L = L1 + L2
        if get_numpy:
            return L.detach().cpu().numpy()
        else:
            return L

    def _estimate_q_from_weights(self, W1_t, W2_t):
        if self.verbose:
            print("Estimating Q of the network")
        Q1_t = []
        Q2_t = []
        task_id = []
        for t_index in range(self.n_steps):
            section = np.floor(t_index / self.change_tasks_every)
            current_dataset_id = int(section % len(self.dataset.dataset_classes))
            task_id.append(current_dataset_id)
            W1 = W1_t[t_index, :, :]
            W2 = W2_t[t_index, :, :]
            if current_dataset_id == 0:
                Us_1 = self.svd_task1["Us_1"]
                Vs_1 = self.svd_task1["Vs_1"]
            else:
                Us_1 = self.svd_task2["Us_1"]
                Vs_1 = self.svd_task2["Vs_1"]
            Q2 = Us_1 @ W2
            Q1 = W1 @ Vs_1
            Q1_t.append(Q1)
            Q2_t.append(Q2)
        return np.stack(Q1_t, axis=0), np.stack(Q2_t, axis=0), np.array(task_id)

    def get_optimal_loss(self, sol, out_cov, in_out_cov, in_cov):
        loss = 0.5 * (np.trace(out_cov) - np.trace(2 * in_out_cov @ sol) + np.trace(in_cov @ sol.T @ sol))
        return loss

    def sample_optimal_qs(self, task1_svd, task2_cov, n_qs=20):
        all_Qs = []
        last_loss = []
        i = 0
        while True:
            try:
                loss, Q = self.estimate_best_q(task1_svd, task2_cov)
            except:
                continue
            all_Qs.append(Q)
            last_loss.append(np.sum(loss[-1])*self.dt)
            i += 1
            if i == n_qs:
                break
        return np.stack(all_Qs, axis=0), np.array(last_loss)


if __name__ == "__main__":
    verbose = True
    results_path = "../../results/task_switch_main/slow_switch_run0_AffineCorrelatedGaussian_27-12-2022_20-36-08-225"
    qa = QAnalysis(results_path=results_path, verbose=verbose)
    loss_trajectories, Q12 = qa.estimate_best_q(qa.svd_task1, qa.cov_matrix_task2)
    print("First Q12", Q12)
    print("row_mean", torch.mean(Q12, dim=0))
    print("col_mean", torch.mean(Q12, dim=1))
    print("all_mean", torch.mean(Q12, dim=(0, 1)))
    loss_trajectories, Q12 = qa.estimate_best_q(qa.svd_task1, qa.cov_matrix_task2)
    print("Second Q12", Q12)
    print("row_mean", torch.mean(Q12, dim=0))
    print("col_mean", torch.mean(Q12, dim=1))
    print("all_mean", torch.mean(Q12, dim=(0, 1)))
    loss_trajectories, Q12 = qa.estimate_best_q(qa.svd_task1, qa.cov_matrix_task2)
    print("Third Q12", Q12)
    print("row_mean", torch.mean(Q12, dim=0))
    print("col_mean", torch.mean(Q12, dim=1))
    print("all_mean", torch.mean(Q12, dim=(0, 1)))
    print("debug")