import torch
from metamod.control import LinearNetEq
import numpy as np


class LinearNetTaskEngEq(LinearNetEq):

    def __init__(self, in_out_cov, in_cov, out_cov, expected_x, expected_y, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 task_output_index=(), task_input_index=(), engagement_coef=None):

        if len(task_output_index) != engagement_coef.shape[1]:
            raise Exception("task index not equal to number of eng coef")

        # List of expected values, as well as every cov matrix in the input
        self.tasks_expected_x = expected_x
        self.tasks_expected_y = expected_y
        self.task_output_index = task_output_index
        self.task_input_index = task_input_index
        self.engagement_coef = engagement_coef
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

        for i, expected_x in enumerate(self.tasks_expected_x):
            for j, expected_y in enumerate(self.tasks_expected_y):
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