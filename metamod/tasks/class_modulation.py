from metamod.tasks import MNIST
import numpy as np
import matplotlib.pyplot as plt


class MNISTClassMod(MNIST):

    def __init__(self, batch_size=32, subset=(1, 7, 3), new_shape=None, affine_data=True,
                 training_mode=True, class_proportions=None):

        self.class_proportions = class_proportions
        self.reset_batch_sampling()
        super().__init__(batch_size, subset, new_shape, affine_data, training_mode)
        self.separate_data_per_class()

    def reset_batch_sampling(self):
        self._batch_id_t = 0

    def separate_data_per_class(self):
        self.input_per_class = []
        self.output_per_class = []
        all_c_idx = np.argmax(self.train_label_data, axis=1)
        for c in range(len(self.subsets)):
            c_idx = np.where(c == all_c_idx)[0]
            self.input_per_class.append(self.train_input_data[c_idx, :])
            self.output_per_class.append(self.train_label_data[c_idx, :])

    def sample_batch(self, training=None):
        x = []
        y = []
        for c in range(len(self.subsets)):
            c_x_data = self.input_per_class[c]
            c_y_data = self.output_per_class[c]
            effective_batch_size = int(self.batch_size*self.class_proportions[self._batch_id_t, c]/len(self.subsets))
            batch_idx = np.random.choice(np.arange(len(c_y_data)), size=effective_batch_size, replace=False)
            x.append(c_x_data[batch_idx, :])
            y.append(c_y_data[batch_idx])
        x = np.concatenate(x, axis=0)
        # print(x.shape)
        y = np.concatenate(y, axis=0)
        self._batch_id_t += 1
        return x, y


if __name__ == "__main__":
    batch_size = 34
    new_shape = (10, 10)
    subset = (0, 1, 2, 3)
    class_proportions = np.ones((20, 4))
    data = MNISTClassMod(batch_size=batch_size, subset=subset, new_shape=new_shape,
                         class_proportions=class_proportions)
    batch_x, batch_y = data.sample_batch()
    print(batch_x.shape, batch_y.shape)

    f, ax = plt.subplots(2, 10, figsize=(80, 14))
    ax = ax.flatten()
    for i in range(20):
        title = batch_y[i, :]
        img = batch_x[i, :-1].reshape(new_shape)
        ax[i].imshow(img)
        ax[i].set_title(str(title))
    plt.show()

    input_corr, output_corr, input_output_corr, expected_y, expected_x = data.get_correlation_matrix()
    print(input_corr.shape, output_corr.shape, input_output_corr.shape, expected_y.shape, expected_x.shape)