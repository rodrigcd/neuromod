import torch
from .base_network import BaseNetwork


class LinearNet(BaseNetwork):

    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, reg_coef=0.1, W1_0=None, W2_0=None,
                 intrinsic_noise=1.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.reg_coef = reg_coef
        self.dtype = torch.float32
        self.intrinsic_noise = intrinsic_noise

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
        self.W1 = self.W1_0.clone().detach().requires_grad_(True)
        self.W2 = self.W2_0.clone().detach().requires_grad_(True)

    def reset_weights(self):
        self.W1 = self.W1_0.clone().detach().requires_grad_(True)
        self.W2 = self.W2_0.clone().detach().requires_grad_(True)

    def forward(self, x):
        x_tensor = torch.from_numpy(x.T).type(self.dtype).to(self.device)
        y_hat = self.W2 @ self.W1 @ x_tensor
        return y_hat + torch.normal(mean=0, std=self.intrinsic_noise, size=y_hat.shape).to(self.device)

    def controlled_forward(self, x, g1_tilda, g2_tilda):
        x_tensor = torch.from_numpy(x.T).type(self.dtype).to(self.device)
        y_hat = (self.W2 * g2_tilda) @ (self.W1 * g1_tilda) @ x_tensor
        return y_hat + torch.normal(mean=0, std=self.intrinsic_noise, size=y_hat.shape).to(self.device)

    def train_step(self, x, y, g1_tilda=None, g2_tilda=None):
        if g1_tilda is None:
            y_pred = self.forward(x)
        else:
            y_pred = self.controlled_forward(x, g1_tilda=g1_tilda, g2_tilda=g2_tilda)
        y_target = torch.from_numpy(y.T).to(self.device)
        loss1 = torch.sum((y_pred - y_target)**2)/(2.0*y.shape[0])
        loss2 = (self.reg_coef/2.0)*(torch.sum(self.W1**2) + torch.sum(self.W2**2))
        loss = loss1 + loss2
        loss.backward()
        self.update_rule()
        return loss

    def update_rule(self):
        with torch.no_grad():
            self.W1 -= self.learning_rate*self.W1.grad
            self.W2 -= self.learning_rate*self.W2.grad
            self.W1.grad = None
            self.W2.grad = None