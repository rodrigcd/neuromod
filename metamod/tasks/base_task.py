import numpy as np


class BaseTask(object):

    def sample_batch(self):
        pass

    def get_correlation_matrix(self):
        return [[] for i in range(5)]

    def get_linear_regression_solution(self, reg_coef=0.0):
        correlation_matrices = self.get_correlation_matrix()
        input_corr = correlation_matrices[0]
        input_output_corr = correlation_matrices[2]
        linear_mapping = input_output_corr.T @ np.linalg.inv(input_corr)
        return linear_mapping