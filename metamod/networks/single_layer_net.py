import torch
from .base_network import BaseNetwork


class SingleLayerNet(BaseNetwork):

    def __init__(self, input_dim, output_dim, learning_rate=1e-5, reg_coef=0.1, W_0=None,
                 intrinsic_noise=1.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.reg_coef = reg_coef
        self.dtype = torch.float32
        self.intrinsic_noise = intrinsic_noise

        if W_0 is None:
            self.W_0 = torch.normal(mean=0.0, std=0.01, size=(self.output_dim, self.input_dim), requires_grad=False,
                                     device=self.device)
        else:
            self.W_0 = torch.from_numpy(W_0).requires_grad_(False).type(self.dtype).to(self.device)
        self.W = self.W_0.clone().detach().requires_grad_(True)

    def reset_weights(self):
        self.W = self.W_0.clone().detach().requires_grad_(True)

    def forward(self, x):
        x_tensor = torch.from_numpy(x.T).type(self.dtype).to(self.device)
        y_hat = self.W @ x_tensor
        return y_hat + torch.normal(mean=0, std=self.intrinsic_noise, size=y_hat.shape).to(self.device)

    def controlled_forward(self, x, g_tilda):
        x_tensor = torch.from_numpy(x.T).type(self.dtype).to(self.device)
        y_hat = (self.W * g_tilda) @ x_tensor
        return y_hat + torch.normal(mean=0, std=self.intrinsic_noise, size=y_hat.shape).to(self.device)

    def train_step(self, x, y, g_tilda=None):
        if g_tilda is None:
            y_pred = self.forward(x)
        else:
            y_pred = self.controlled_forward(x, g_tilda=g_tilda)
        y_target = torch.from_numpy(y.T).to(self.device)
        loss1 = torch.sum((y_pred - y_target)**2)/(2.0*y.shape[0])
        loss2 = (self.reg_coef/2.0)*(torch.sum(self.W**2))
        loss = loss1 + loss2
        loss.backward()
        self.update_rule()
        return loss

    def update_rule(self):
        with torch.no_grad():
            self.W -= self.learning_rate*self.W.grad
            self.W.grad = None