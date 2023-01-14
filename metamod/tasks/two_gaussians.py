import numpy as np
from metamod.tasks import BaseTask


class TwoGaussians(BaseTask):

    def __init__(self, mu=2, std=2, batch_size=32):
        self.mu = mu
        self.std = std
        self.batch_size = batch_size
        self.input_dim = 1
        self.output_dim = 1

    def sample_batch(self):
        y = 1-2*np.random.binomial(1, 0.5, size=self.batch_size)
        x = np.random.normal(loc=y*self.mu, scale=self.std)

        return x[:, np.newaxis], y[:, np.newaxis]

    def get_correlation_matrix(self):
        expected_x = np.array([0, ])
        expected_y = np.array([0, ])
        input_corr = np.array([self.mu**2 + self.std**2, ])[np.newaxis, :]
        input_output_corr = np.array([self.mu, ])[np.newaxis, :]
        output_corr = np.array([1 ,])[np.newaxis, :]
        return input_corr, output_corr, input_output_corr, expected_y, expected_x


if __name__ == "__main__":
    batch_size = 100000
    data = TwoGaussians(batch_size=batch_size)
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