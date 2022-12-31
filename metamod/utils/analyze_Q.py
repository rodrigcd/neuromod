import numpy as np
from metamod.utils import ResultsManager
import torch


class QAnalysis(object):

    def __init__(self, results_path, optimizing_window=100, verbose=False, q_iters=100, q_opt_lr=1.0):
        self.verbose = verbose
        self.results_path = results_path
        self.optimizing_window = optimizing_window
        self.Q_iters = q_iters
        self.Q_opt_lr = q_opt_lr
        self.results = ResultsManager(results_path, verbose=verbose)
        self.net_eq = self.results.params["equation_params"]["solver"]
        self.dataset = self.results.params["dataset_params"]["dataset"]
        self.change_tasks_every = self.dataset.change_tasks_every
        self.n_steps = self.results.params["equation_params"]["n_steps"]
        self._compute_regression_solution()

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

    def estimate_best_q(self, task1_svd, task2_cov):
        # Move variables to device
        device = self.net_eq.device
        dtype = self.net_eq.dtype
        gpu_svd = {}
        for key, value in task1_svd.items():
            gpu_svd[key] = torch.tensor(value, dtype=dtype, device=device, requires_grad=False)
        gpu_cov = []
        for value in task2_cov:
            gpu_cov.append(torch.tensor(value, dtype=dtype, device=device, requires_grad=False))

        # Initialize Q
        hidden_units = self.net_eq.hidden_dim
        Q_matrix = torch.normal(mean=0, std=0.01, size=(hidden_units, task1_svd["S"].shape[1]),
                                requires_grad=True, device=device, dtype=dtype)
        Q_inverse = torch.linalg.pinv(Q_matrix)

        # Build weights
        W2_task_init = gpu_svd["U"] @ torch.sqrt(gpu_svd["S"]) @ Q_inverse
        W1_task_init = Q_matrix @ torch.sqrt(gpu_svd["S_prime"]).T @ gpu_svd["V_T"]

        # Replace init weights and optimizing period
        self.net_eq.W1 = W1_task_init
        self.net_eq.W2 = W2_task_init
        self.net_eq.time_span = np.arange(self.optimizing_window) * self.net_eq.learning_rate
        dt = self.net_eq.time_span[1] - self.net_eq.time_span[0]

        # Adjust covariance for second task
        self.net_eq.in_out_cov = gpu_cov[2]
        self.net_eq.in_cov = gpu_cov[0]
        self.net_eq.out_cov = gpu_cov[1]
        self.net_eq.n_steps = self.optimizing_window
        self.net_eq.in_out_cov_list = [task2_cov[2], ]
        self.net_eq.in_cov_list = [task2_cov[0], ]
        self.net_eq.out_cov_list = [task2_cov[1], ]
        self.net_eq.change_task_every = self.optimizing_window + 100
        self.net_eq.current_dataset_id = 0
        self.net_eq._generate_cov_matrices()

        # Run loop
        loss_trajectories = []
        for i in range(self.Q_iters):
            W1_t, W2_t = self.net_eq.get_weights(self.net_eq.time_span)
            L_t = self.net_eq.get_loss_function(W1=W1_t, W2=W2_t)
            loss_trajectories.append(L_t.detach().cpu().numpy())
            print("Loss integral:", np.sum(loss_trajectories)*dt)
            loss_integral = torch.sum(L_t)*dt
            loss_integral.backward()
            with torch.no_grad():
                Q_matrix += self.Q_opt_lr * Q_matrix.grad
            Q_inverse = torch.linalg.pinv(Q_matrix)
            # Build weights
            W2_task_init = gpu_svd["U"] @ torch.sqrt(gpu_svd["S"]) @ Q_inverse
            W1_task_init = Q_matrix @ torch.sqrt(gpu_svd["S_prime"]).T @ gpu_svd["V_T"]
            self.net_eq.W1 = W1_task_init
            self.net_eq.W2 = W2_task_init

        return loss_trajectories, Q_matrix

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


if __name__ == "__main__":
    verbose = True
    results_path = "../../results/task_switch_main/slow_switch_run0_AffineCorrelatedGaussian_27-12-2022_20-36-08-225"
    qa = QAnalysis(results_path=results_path, verbose=verbose)
    loss_trajectories, Q = qa.estimate_best_q(qa.svd_task1, qa.cov_matrix_task2)
    print("debug")