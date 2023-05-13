from torchvision import datasets
import matplotlib.pyplot as plt
from metamod.tasks import BaseTask
import numpy as np
from skimage.transform import resize
import metamod
import os
import torch


class MNIST(BaseTask):

    def __init__(self, batch_size=32, subset=(1, 7, 3), new_shape=None, affine_data=True,
                 training_mode=True, tensor_mode=False):
        self.batch_size = batch_size
        self.new_shape = new_shape
        self.affine_data = affine_data
        self.training_mode = training_mode
        self.tensor_mode = tensor_mode
        mnist_data_path = os.path.join(metamod.__path__[0], "tasks/data_mnist")

        self.training_data = datasets.MNIST(
            root=mnist_data_path,
            train=True,
            download=True
        )
        self.test_data = datasets.MNIST(
            root=mnist_data_path,
            train=False,
            download=True
        )
        self.subsets = subset
        self._select_subset()
        self._transform_data()

    def _select_subset(self):
        self.train_input_data = self.training_data.data.detach().cpu().numpy()
        self.train_label_data = self.training_data.targets.detach().cpu().numpy()

        self.test_input_data = self.test_data.data.detach().cpu().numpy()
        self.test_label_data = self.test_data.targets.detach().cpu().numpy()

        if self.subsets is None:
            return
        else:
            train_idx = []
            test_idx = []
            for ss in self.subsets:
                train_idx.append(np.where(self.train_label_data == ss)[0])
                test_idx.append(np.where(self.test_label_data == ss)[0])
            train_idx = np.concatenate(train_idx)
            test_idx = np.concatenate(test_idx)

        self.train_input_data = self.train_input_data[train_idx, ...]
        self.train_label_data = self.train_label_data[train_idx]
        self.test_input_data = self.test_input_data[test_idx, ...]
        self.test_label_data = self.test_label_data[test_idx]

    def _transform_data(self):
        if self.new_shape is not None:
            transformed_train = []
            transformed_test = []
            for i in range(len(self.train_label_data)):
                img = self.train_input_data[i, :, :]
                transformed_train.append(resize(img, self.new_shape))
            for i in range(len(self.test_label_data)):
                img = self.test_input_data[i, :, :]
                transformed_test.append(resize(img, self.new_shape))
            self.train_input_data = np.stack(transformed_train, axis=0)
            self.test_input_data = np.stack(transformed_test, axis=0)

        # Flatten
        self.train_input_data = np.reshape(self.train_input_data, newshape=(len(self.train_label_data), -1))
        self.test_input_data = np.reshape(self.test_input_data, newshape=(len(self.test_label_data), -1))

        # On hot encoding
        train_labels = np.zeros(shape=(len(self.train_label_data), len(self.subsets)))
        test_labels = np.zeros(shape=(len(self.test_label_data), len(self.subsets)))
        for i, ss in enumerate(self.subsets):
            train_idx = np.where(self.train_label_data == ss)[0]
            test_idx = np.where(self.test_label_data == ss)[0]
            train_labels[train_idx, i] = 1
            test_labels[test_idx, i] = 1
        self.train_label_data = train_labels
        self.test_label_data = test_labels

        self.n_train = len(self.train_label_data)
        self.n_test = len(self.test_label_data)

        # Affine data
        if self.affine_data:
            self.train_input_data = np.concatenate([self.train_input_data, np.ones(shape=(self.n_train, 1))], axis=1)
            self.test_input_data = np.concatenate([self.test_input_data, np.ones(shape=(self.n_test, 1))], axis=1)

        self.input_dim = self.train_input_data.shape[1]
        self.output_dim = self.test_label_data.shape[1]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        self.tensor_train_input_data = torch.tensor(self.train_input_data, device=self.device, dtype=self.dtype)
        self.tensor_train_label_data = torch.tensor(self.train_label_data, device=self.device, dtype=self.dtype)
        self.tensor_test_input_data = torch.tensor(self.test_input_data, device=self.device, dtype=self.dtype)
        self.tensor_test_label_data = torch.tensor(self.test_label_data, device=self.device, dtype=self.dtype)

    def sample_batch(self, training=None, tensor_mode=False):
        if training is None:
            training = self.training_mode
        if training:
            batch_idx = np.random.choice(np.arange(len(self.train_label_data)), size=self.batch_size, replace=True)
            if tensor_mode:
                img = self.tensor_train_input_data[batch_idx, :]
                label = self.tensor_train_label_data[batch_idx]
            else:
                img = self.train_input_data[batch_idx, :]
                label = self.train_label_data[batch_idx]
        else:
            batch_idx = np.random.choice(np.arange(len(self.test_label_data)), size=self.batch_size, replace=True)
            if tensor_mode:
                img = self.tensor_test_input_data[batch_idx, :]
                label = self.tensor_test_label_data[batch_idx]
            else:
                img = self.test_input_data[batch_idx, :]
                label = self.test_label_data[batch_idx]
        return img, label

    def get_correlation_matrix(self, training=None):
        if training is None:
            training = self.training_mode
        if training:
            input_data = self.train_input_data
            label_data = self.train_label_data
            n_samples = self.n_train
        else:
            input_data = self.test_input_data
            label_data = self.test_label_data
            n_samples = self.n_test

        input_corr = (input_data.T @ input_data)/n_samples
        output_corr = (label_data.T @ label_data)/n_samples
        input_output_corr = (input_data.T @ label_data)/n_samples

        expected_x = np.mean(input_data, axis=0)
        expected_y = np.mean(label_data, axis=0)

        return input_corr, output_corr, input_output_corr, expected_y, expected_x


if __name__ == "__main__":
    batch_size = 34
    new_shape = (5, 5)
    subset = (0, 1, 2, 3)
    data = MNIST(batch_size=batch_size, subset=subset, new_shape=new_shape)
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
