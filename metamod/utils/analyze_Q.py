import numpy as np
from metamod.utils import ResultsManager


class QAnalysis(object):

    def __init__(self, results_path, verbose=False):
        self.verbose = verbose
        self.results_path = results_path
        self.results = ResultsManager(results_path, verbose=verbose)
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

        self.baseline_Q1, self.baseline_Q2, self.task_id_t = self._estimate_Q_from_weights(self.baseline_W1,
                                                                                           self.baseline_W2)
        self.control_Q1, self.control_Q2, _ = self._estimate_Q_from_weights(self.control_W1,
                                                                            self.control_W2)

    def estimate_best_Q(self):
        pass

    def _estimate_Q_from_weights(self, W1_t, W2_t):
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
    qa = QAnalysis(results_path, verbose)
