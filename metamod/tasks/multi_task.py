from metamod.tasks import BaseTask
import numpy as np


class MultiTask(BaseTask):

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

        self.input_dim = self.input_dim_per_task[0]
        self.output_dim = self.output_dim_per_task[0]

    def sample_batch(self):
        batch_x = []
        batch_y = []
        for dataset in self.datasets:
            x, y = dataset.sample_batch()
            batch_x.append(x)
            batch_y.append(y)
        return batch_x, batch_y

    def get_correlation_matrix(self, training=None):
        input_corr = []
        output_corr = []
        input_output_corr = []
        expected_y = []
        expected_x = []
        for i, dataset in enumerate(self.datasets):
            input_corr_i, output_corr_i, input_output_corr_i, expected_y_i, expected_x_i = dataset.get_correlation_matrix(training=training)
            input_corr.append(input_corr_i)
            output_corr.append(output_corr_i)
            input_output_corr.append(input_output_corr_i)
            expected_x.append(expected_x_i)
            expected_y.append(expected_y_i)
        return input_corr, output_corr, input_output_corr, expected_y, expected_x
