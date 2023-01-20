from metamod.networks import LinearTaskEngNet, LinearNet
import torch


class LinearCatPropNet(LinearNet):

    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, reg_coef=0.1, W1_0=None, W2_0=None,
                 intrinsic_noise=1.0):
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, reg_coef, W1_0, W2_0,
                         intrinsic_noise)

    def train_step(self, x, y, g1_tilda=None, g2_tilda=None, engagement_coef=None):
        # Engagement coefficient should be equal to the number of tasks
        if g1_tilda is None:
            y_pred = self.forward(x)
        else:
            y_pred = self.controlled_forward(x, g1_tilda=g1_tilda, g2_tilda=g2_tilda)

        if g1_tilda is None:
            y_pred = self.forward(x)
        else:
            y_pred = self.controlled_forward(x, g1_tilda=g1_tilda, g2_tilda=g2_tilda)
        y_target = torch.from_numpy(y.T).type(self.dtype).to(self.device)
        loss1 = torch.sum((y_pred - y_target)**2)/(2.0*y.shape[0])
        loss2 = (self.reg_coef/2.0)*(torch.sum(self.W1**2) + torch.sum(self.W2**2))
        real_loss = loss1 + loss2

        if engagement_coef is None:
            real_loss.backward()
            self.update_rule()
        else:
            diagonal_coef = torch.diag(engagement_coef)
            error_signal = diagonal_coef @ (y_pred - y_target)
            loss1_ = torch.sum(error_signal**2)/(2.0*y.shape[0])
            loss2_ = (self.reg_coef / 2.0) * (torch.sum(self.W1 ** 2) + torch.sum(self.W2 ** 2))
            loss_ = loss1_ + loss2_
            loss_.backward()
            self.update_rule()

        return real_loss
