import torch
import numpy as np
from metamod.value_estimation import SigmoidEstimator
from metamod.utils import detach_torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    starting_params = np.array([1, -3, 1, -1])

    sigmoid_estimator = SigmoidEstimator(starting_params=starting_params + np.random.normal(0, 0.5, size=len(starting_params)))
    time_array = torch.linspace(0, 10, 100, dtype=sigmoid_estimator.dtype, device=sigmoid_estimator.device)
    sigmoid = torch.nn.Sigmoid()
    target = sigmoid(time_array*starting_params[0] + starting_params[1]) * starting_params[2] + starting_params[3]

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(detach_torch(time_array), detach_torch(target), "-o", label="target")
    plt.show()
