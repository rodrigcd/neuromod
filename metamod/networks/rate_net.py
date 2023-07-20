import torch
from .base_network import BaseNetwork
from .linear_net import LinearNet


class LRLinearNet(LinearNet):

    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, reg_coef=0.1, W1_0=None, W2_0=None,
                 intrinsic_noise=1.0):
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, reg_coef, W1_0, W2_0, intrinsic_noise)

    def train_step(self, x, y, g1_tilda=None, g2_tilda=None, opt_lr=None):
        if g1_tilda is None:
            y_pred = self.forward(x)
        else:
            y_pred = self.controlled_forward(x, g1_tilda=g1_tilda, g2_tilda=g2_tilda)
        y_target = torch.from_numpy(y.T).type(self.dtype).to(self.device)
        loss1 = torch.sum((y_pred - y_target)**2)/(2.0*y.shape[0])
        loss2 = (self.reg_coef/2.0)*(torch.sum(self.W1**2) + torch.sum(self.W2**2))
        loss = loss1 + loss2
        loss.backward()
        self.update_rule(opt_lr)
        return loss

    def update_rule(self, opt_lr=None):
        if opt_lr is None:
            opt_lr = self.learning_rate
        with torch.no_grad():
            self.W1 -= opt_lr * self.W1.grad
            self.W2 -= opt_lr * self.W2.grad
            self.W1.grad = None
            self.W2.grad = None