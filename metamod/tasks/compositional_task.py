from metamod.tasks import BaseTask, AffineCorrelatedGaussian
import numpy as np


class CompositionOfTasks(BaseTask):

    def __init__(self, dataset_classes=(), dataset_list_params=()):
        if len(dataset_classes) <= 0:
            raise Exception("number of datasets less than 1")

        self.dataset_classes = dataset_classes
        self.dataset_params = dataset_list_params

        self.datasets = []
        self.input_dim_per_task = []
        self.output_dim_per_task = []
        for i, dataset_class in enumerate(self.dataset_classes):
            self.datasets.append(dataset_class(**self.dataset_params[i]))
            self.input_dim_per_task.append(self.datasets[-1].input_dim)
            self.output_dim_per_task.append(self.datasets[-1].output_dim)

        task_output_limits = np.concatenate([[0, ], np.cumsum(self.output_dim_per_task)])
        task_input_limits = np.concatenate([[0, ], np.cumsum(self.input_dim_per_task)])
        self.task_output_index = []
        self.task_input_index = []
        for i in range(len(self.datasets)):
            self.task_output_index.append((task_output_limits[i], task_output_limits[i + 1]))
            self.task_input_index.append((task_input_limits[i], task_input_limits[i + 1]))

        self.input_dim = np.sum(self.input_dim_per_task).astype(int)
        self.output_dim = np.sum(self.output_dim_per_task).astype(int)

    def sample_batch(self):
        batch_x = []
        batch_y = []
        for dataset in self.datasets:
            x, y = dataset.sample_batch()
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.concatenate(batch_x, axis=1)
        batch_y = np.concatenate(batch_y, axis=1)
        return batch_x, batch_y

    def get_correlation_matrix(self):
        input_corr = []
        output_corr = []
        input_output_corr = []
        expected_y = []
        expected_x = []
        for i, dataset in enumerate(self.datasets):
            input_corr_i, output_corr_i, input_output_corr_i, expected_y_i, expected_x_i = dataset.get_correlation_matrix()
            input_corr.append(input_corr_i)
            output_corr.append(output_corr_i)
            input_output_corr.append(input_output_corr_i)
            expected_x.append(expected_x_i)
            expected_y.append(expected_y_i)
        return input_corr, output_corr, input_output_corr, expected_y, expected_x
