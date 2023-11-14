import numpy as np
import torch
from metamod.networks import BaseNetwork


class FullGradNet(BaseNetwork):

    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, reg_coef=0.1, W1_0=None, W2_0=None,
                 intrinsic_noise=1.0, keep_grads=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.reg_coef = reg_coef
        self.dtype = torch.float32
        self.intrinsic_noise = intrinsic_noise
        self.keep_grads = True
        self.reset_weights(W1_0, W2_0)

    def reset_weights(self, W1_0=None, W2_0=None):
        if W1_0 is None:
            self.W1_0 = torch.normal(mean=0.0, std=0.01, size=(self.hidden_dim, self.input_dim), requires_grad=False,
                                     device=self.device)
        else:
            self.W1_0 = torch.from_numpy(W1_0).requires_grad_(False).type(self.dtype).to(self.device)
        if W2_0 is None:
            self.W2_0 = torch.normal(mean=0.0, std=0.01, size=(self.output_dim, self.hidden_dim), requires_grad=False,
                                     device=self.device)
        else:
            self.W2_0 = torch.from_numpy(W2_0).requires_grad_(False).type(self.dtype).to(self.device)
        self.W1 = self.W1_0.clone().detach().requires_grad_(False)
        self.W2 = self.W2_0.clone().detach().requires_grad_(False)

    def hidden_layer(self, x_tensor, g1_tilda=None):
        if g1_tilda is None:
            linear_hidden = self.W1 @ x_tensor
        else:
            linear_hidden = (self.W1 * g1_tilda) @ x_tensor
        non_linear_hidden = torch.tanh(linear_hidden)
        return non_linear_hidden, linear_hidden

    def hidden_der(self, func_in):
        return 1 - torch.tanh(func_in)

    def forward(self, x):
        x_tensor = torch.from_numpy(x.T).type(self.dtype).to(self.device)
        non_linear_hidden, linear_hidden = self.hidden_layer(x_tensor)
        y_hat = self.W2 @ non_linear_hidden
        pred = y_hat + torch.normal(mean=0, std=self.intrinsic_noise, size=y_hat.shape).to(self.device)
        return pred

    def controlled_forward(self, x, g1_tilda, g2_tilda):
        x_tensor = torch.from_numpy(x.T).type(self.dtype).to(self.device)
        non_linear_hidden, linear_hidden = self.hidden_layer(x_tensor, g1_tilda)
        y_hat = (self.W2 * g2_tilda) @ non_linear_hidden
        pred = y_hat + torch.normal(mean=0, std=self.intrinsic_noise, size=y_hat.shape).to(self.device)
        return pred

    def train_step(self, x, y, g1_tilda=None, g2_tilda=None):
        x_tensor = torch.from_numpy(x.T).type(self.dtype).to(self.device)
        y_target = torch.from_numpy(y.T).type(self.dtype).to(self.device)

        if g1_tilda is None:
            non_linear_hidden, linear_hidden = self.hidden_layer(x_tensor=x_tensor)
            y_pred = self.W2 @ non_linear_hidden
        else:
            non_linear_hidden, linear_hidden = self.hidden_layer(x_tensor=x_tensor, g1_tilda=g1_tilda)
            y_pred = (self.W2 * g2_tilda) @ non_linear_hidden

        loss1 = torch.sum((y_pred - y_target)**2)/(2.0*y.shape[0])
        loss2 = (self.reg_coef/2.0)*(torch.sum(self.W1**2) + torch.sum(self.W2**2))
        loss = loss1 + loss2

        y_target_v = y_target.T[:, :, None]
        x_tensor_T = x_tensor.T[:, None, :]

        # dW2/dt
        non_linear_T = non_linear_hidden.T[:, None, :]
        non_linear = non_linear_hidden.T[:, :, None]
        dW2 = y_target_v @ non_linear_T - self.W2[None, :, :] @ non_linear @ non_linear_T - self.reg_coef * self.W2[None, :, :]

        # dW1/dt
        linear = linear_hidden.T[:, :]
        diag_f = torch.diag_embed(self.hidden_der(linear), dim1=-2, dim2=-1)
        dW1 = diag_f @ self.W2.T[None, :, :] @ y_target_v @ x_tensor_T \
              - torch.transpose((self.W2[None, :, :] @ diag_f), dim0=-1, dim1=-2) @ (self.W2[None, :, :] @ non_linear) @ x_tensor_T \
              - self.reg_coef * self.W1[None, :, :]

        self.W1 = self.W1 + self.learning_rate * torch.mean(dW1, dim=0)
        self.W2 = self.W2 + self.learning_rate * torch.mean(dW2, dim=0)

        return loss


class CatEngGradNet(FullGradNet):
    # TODO: Rewrite train step
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, reg_coef=0.1, W1_0=None, W2_0=None,
                 intrinsic_noise=1.0, keep_grads=False):
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, reg_coef, W1_0, W2_0, intrinsic_noise, keep_grads)
