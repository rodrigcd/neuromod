import numpy as np
from .base_task import BaseTask


class AffineCorrelatedGaussian(BaseTask):

    def __init__(self, mu_vec=(2.0, 2.0), sigma_vec=(1.0, 1.0), dependence_parameter=0.5, batch_size=128):
        self.mu = mu_vec
        self.sigma = sigma_vec
        self.dep_param = dependence_parameter
        self.batch_size = batch_size
        self.input_dim = 3
        self.output_dim = 2

    def sample_batch(self):
        y1 = 1-2*np.random.binomial(1, 0.5, size=self.batch_size)
        y2 = (1-2*np.random.binomial(1, self.dep_param, size=self.batch_size))*y1

        x1 = np.random.normal(loc=y1*self.mu[0], scale=self.sigma[0])
        x2 = np.random.normal(loc=y2*self.mu[1], scale=self.sigma[1])
        x3 = np.ones(shape=self.batch_size)

        return np.stack([x1, x2, x3], axis=1), np.stack([y1, y2], axis=1)

    def get_correlation_matrix(self):
        output_corr = np.array([[1, 1-2*self.dep_param],
                                [1-2*self.dep_param, 1]])
        input_output_corr = np.array([[self.mu[0], self.mu[0]*(1-2*self.dep_param)],
                                      [self.mu[1]*(1-2*self.dep_param), self.mu[1]],
                                      [0, 0]])
        input_corr = np.array([[self.mu[0]**2 + self.sigma[0]**2, self.mu[0]*self.mu[1]*(1-2*self.dep_param), 0],
                              [self.mu[0]*self.mu[1]*(1-2*self.dep_param), self.mu[1]**2 + self.sigma[1]**2, 0],
                              [0, 0, 1]])
        expected_y = np.array([0, 0])
        expected_x = np.array([0, 0, 1])
        return input_corr, output_corr, input_output_corr, expected_y, expected_x


if __name__ == "__main__":
    batch_size = 100000
    data = AffineCorrelatedGaussian(dependence_parameter=0.2, batch_size=batch_size)
    batch_x, batch_y = data.sample_batch()

    print(batch_x.shape, batch_y.shape)
    input_corr, output_corr, input_output_corr, expected_y, expected_x = data.get_correlation_matrix()

    est_input_corr = 0
    for i in range(batch_size):
        est_input_corr += batch_x[i, :, np.newaxis] @ batch_x[i, :, np.newaxis].T
    est_input_corr = est_input_corr / batch_size

    est_output_corr = 0
    for i in range(batch_size):
        est_output_corr += batch_y[i, :, np.newaxis] @ batch_y[i, :, np.newaxis].T
    est_output_corr = est_output_corr / batch_size

    est_input_output_corr = 0
    for i in range(batch_size):
        est_input_output_corr += batch_x[i, :, np.newaxis] @ batch_y[i, :, np.newaxis].T
    est_input_output_corr = est_input_output_corr / batch_size

    est_expected_x = 0
    for i in range(batch_size):
        est_expected_x += batch_x[i, :]
    est_expected_x = est_expected_x / batch_size

    est_expected_y = 0
    for i in range(batch_size):
        est_expected_y += batch_y[i, :]
    est_expected_y = est_expected_y / batch_size

    print("input_corr \n", input_corr)
    print("est_input_corr \n", est_input_corr)

    print("output_corr \n", output_corr)
    print("est_output_corr \n", est_output_corr)

    print("input_output_corr", input_output_corr)
    print("est_input_output_corr", est_input_output_corr)

    print("expected_x", expected_x)
    print("est_expected_x", est_expected_x)

    print("expected_y", expected_y)
    print("est_expected_y", est_expected_y)