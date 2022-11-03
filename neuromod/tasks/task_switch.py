from neuromod.tasks import BaseTask, AffineCorrelatedGaussian

class TaskSwitch(BaseTask):

    def __init__(self, dataset_classes=(), dataset_list_params=(), change_tasks_every=200):
        if len(dataset_classes) == 0:
            dataset_classes = (AffineCorrelatedGaussian, AffineCorrelatedGaussian)
            dataset1_params = {"mu_vec": (3.0, 1.0), "sigma_vec": (1.0, 1.0), "dependence_parameter": 0.8,
                               "batch_size": 512}
            dataset2_params = {"mu_vec": (-2.0, 2.0), "sigma_vec": (1.0, 1.0), "dependence_parameter": 0.2,
                               "batch_size": 512}
            dataset_list_params = (dataset1_params, dataset2_params)

        self.dataset_classes = dataset_classes
        self.dataset_params = dataset_list_params
        self.change_tasks_every = change_tasks_every
        self.batch_iter = 0

        self.datasets = []
        for i, dataset_class in enumerate(self.dataset_classes):
            self.datasets.append(dataset_class(**self.dataset_params[i]))

        self.current_dataset_id = 0

    def reset(self):
        self.batch_iter = 0

    def sample_batch(self):
        current_dataset = self.datasets[self.current_dataset_id]
        batch_x, batch_y = current_dataset.sample_batch()
        self.batch_iter += 1
        if self.batch_iter % self.change_tasks_every == 0:
            self.current_dataset_id += 1
            if self.current_dataset_id >= len(self.datasets):
                self.current_dataset_id = 0
        return batch_x, batch_y

    def get_correlation_matrix(self):
        input_corr, output_corr, input_output_corr, expected_y, expected_x = [], [], [], [], []

        for dataset in self.datasets:
            stats = dataset.get_correlation_matrix()
            input_corr.append(stats[0])
            output_corr.append(stats[1])
            input_output_corr.append(stats[2])
            expected_y.append(stats[3])
            expected_x.append(stats[4])

        return input_corr, output_corr, input_output_corr, expected_y, expected_x


if __name__ == "__main__":
    batch_size = 1000
    data = TaskSwitch()

    n_iters = 3000
    for i in range(n_iters):
        batch_x, batch_y = data.sample_batch()
        print("iter", i)
        print("current dataset", data.current_dataset_id)

    print(data.get_correlation_matrix())