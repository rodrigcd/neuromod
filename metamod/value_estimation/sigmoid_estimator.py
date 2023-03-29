import numpy as np
import torch
from metamod.utils import detach_torch


class SigmoidEstimator(object):

    def __init__(self, starting_params: np.array, update_periods: int = 20, param_learning_rate: float = 0.01):
        """
        :param starting_params: (time_coef, time_bias, amplitude, bias)
        """
        self.dtype = torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_params = torch.tensor(starting_params, dtype=self.dtype, device=self.device, requires_grad=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.update_periods = update_periods
        self.param_learning_rate = param_learning_rate

    def forward_func(self, t):
        y = self.sigmoid(self.current_params[0] * t + self.current_params[1]) * self.current_params[2] + self.current_params[3]
        return y

    def derivative_func(self, t):
        der_param3 = 1
        der_param2 = self.sigmoid(self.current_params[0] * t + self.current_params[1])
        der_param1 = der_param2 * (1 - der_param2)
        der_param0 = self.current_params[2] * der_param1 * t
        return torch.cat([der_param0, der_param1, der_param2, der_param3], dim=0)

    def update_params(self, t, y):
        for i in range(self.update_periods):
            self.current_params = self.current_params - self.param_learning_rate * self.derivative_func(t) * (self.forward_func(t) - y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    starting_params = np.array([1, 1, 1, -1])

    sigmoid_estimator = SigmoidEstimator(starting_params=starting_params + np.random.normal(0, 0.5, size=len(starting_params)))
    time_array = torch.linspace(0, 10, 100, dtype=sigmoid_estimator.dtype, device=sigmoid_estimator.device)
    sigmoid = torch.nn.Sigmoid()
    target = sigmoid(time_array*starting_params[0] + starting_params[1]) * starting_params[2] + starting_params[3]

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(detach_torch(time_array), detach_torch(target), "-o", label="target")
    plt.show()
