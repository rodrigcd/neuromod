import numpy as np
import matplotlib.pyplot as plt
from metamod.tasks import BaseTask


class SemanticTask(BaseTask):

    def __init__(self, batch_size, h_levels=3, affine_data=True):
        self.batch_size = batch_size
        self.h_level = h_levels
        self.affine_data = affine_data
        self._generate_hierarchy()

    def _generate_hierarchy(self):
        self.input_dim = 2**(self.h_level-1)
        self.output_dim = 2**self.h_level - 1

        self.input_matrix = np.identity(n=self.input_dim)
        self.input_corr = np.identity(n=self.input_dim)/self.input_dim
        self.h_matrix = generate_hierarchy_matrix(self.h_level).T
        self.input_output_corr = self.h_matrix/self.input_dim
        self.output_corr = self.input_output_corr.T @ self.input_output_corr * self.input_dim

    def get_correlation_matrix(self, training=None):
        expected_x = np.mean(self.input_matrix, axis=0)
        expected_y = np.mean(self.h_matrix, axis=0)
        return self.input_corr, self.output_corr, self.input_output_corr, expected_y, expected_x

    def sample_batch(self, training=None):
        batch_idx = np.random.choice(np.arange(self.input_dim), size=self.batch_size, replace=True)
        x = self.input_matrix[batch_idx, :]
        y = self.h_matrix[batch_idx, :]
        return x, y


def generate_hierarchy_matrix(h):
    h = h - 1
    if h < 0:
        raise Exception("Hierarchy < 0")
    else:
        cov_matrix = np.ones((1, 1))
        for i in range(h):
            new_matrix = []
            for j in range(cov_matrix.shape[1]):
                new_matrix.append(cov_matrix[:, j])
                new_matrix.append(cov_matrix[:, j])
            new_matrix = np.stack(new_matrix, axis=1)
            new_matrix = np.concatenate([new_matrix, np.identity(new_matrix.shape[1])])
            cov_matrix = new_matrix
    return cov_matrix


if __name__ == "__main__":
    f, ax = plt.subplots(1, 5, figsize=(6*5, 5))
    ax = ax.flatten()
    for i in range(1, 5):
        print("hierarchy", i)
        matrix = generate_hierarchy_matrix(i)
        ax[i].imshow(matrix, interpolation=None)
        ax[i].set_title(str(matrix.shape)+", h: "+str(i))
    plt.show()

    batch_size = 100000
    data = SemanticTask(batch_size=batch_size, h_levels=3)
    batch_x, batch_y = data.sample_batch()
    print(batch_x.shape, batch_y.shape)
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