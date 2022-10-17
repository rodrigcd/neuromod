import numpy as np
import matplotlib.pyplot as plt


class MultiDimGaussian(object):

    def __init__(self, mu_vec=None, covariance=None, max_mean=2.0, max_std=1.0, batch_size=128, gaussian_dim=4):
        if mu_vec is not None:
            self.mu = mu_vec
            self.input_dim = len(mu_vec) + 1
            self.covariance = covariance
        else:
            self.mu = np.linspace(max_mean/10, max_mean, num=gaussian_dim, endpoint=True)
            self.input_dim = len(self.mu) + 1
            self.covariance = np.identity(self.input_dim-1)*max_std
        self.batch_size = batch_size
        self.output_size = 2

    def sample_batch(self):
        y_sign = 1 - 2 * np.random.binomial(1, 0.5, size=self.batch_size)
        x_normal = np.random.multivariate_normal(np.zeros(self.input_dim-1), cov=self.covariance, size=self.batch_size)
        x = x_normal + self.mu[np.newaxis, :]*y_sign[:, np.newaxis]
        affine_space_x = np.ones(shape=(self.batch_size, 1))
        y = np.zeros((self.batch_size, self.output_size))
        y_hot = (y_sign + 1)/2
        y[np.arange(self.batch_size), y_hot.astype(int)] = 1
        return np.concatenate([x, affine_space_x], axis=1), y

    def get_correlation_matrix(self):
        output_corr = np.array([[0.5, 0],
                                [0, 0.5]])

        input_output_corr = 0.5*(-self.mu[:, np.newaxis] @ np.array([1, 0])[np.newaxis, :] + self.mu[:, np.newaxis] @ np.array([0, 1])[np.newaxis, :])
        input_output_corr = np.concatenate([input_output_corr, 0.5*np.ones((1, input_output_corr.shape[1]))], axis=0)

        input_corr = np.concatenate([self.mu[:, np.newaxis] @ self.mu[np.newaxis, :] + self.covariance, np.zeros(shape=(self.covariance.shape[0], 1))], axis=1)
        input_corr = np.concatenate([input_corr, np.zeros(shape=(1, input_corr.shape[1]))], axis=0)
        input_corr[-1, -1] = 1

        expected_y = np.array([0.5, 0.5])
        expected_x = np.zeros(self.input_dim)
        expected_x[-1] = 1
        return input_corr, output_corr, input_output_corr, expected_y, expected_x


if __name__ == "__main__":
    batch_size = 100000
    data = MultiDimGaussian(batch_size=batch_size)
    print(data.mu, data.covariance)
    batch_x, batch_y = data.sample_batch()

    input_corr, output_corr, input_output_corr, expected_y, expected_x = data.get_correlation_matrix()
    # print(batch_x.shape, batch_y.shape)
    # print(batch_x, batch_y)
    # cs = ["b", "r"]
    # colors = []
    # for i in range(batch_size):
    #     colors.append(cs[np.argmax(batch_y[i, :])])
    #
    # plt.scatter(batch_x[:, 2], batch_x[:, 3], c=colors)
    # plt.show()

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
    print("est_input_corr \n", est_input_corr.round(2))

    print("output_corr \n", output_corr)
    print("est_output_corr \n", est_output_corr.round(2))

    print("input_output_corr", input_output_corr)
    print("est_input_output_corr", est_input_output_corr.round(2))

    print("expected_x", expected_x)
    print("est_expected_x", est_expected_x.round(2))

    print("expected_y", expected_y)
    print("est_expected_y", est_expected_y.round(2))