from metamod.networks import LinearNet
import torch


class LinearTaskEngNet(LinearNet):

    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, reg_coef=0.1, W1_0=None, W2_0=None,
                 intrinsic_noise=1.0, task_output_index=()):
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, reg_coef, W1_0, W2_0,
                         intrinsic_noise)
        self.task_output_index = task_output_index

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
        y_target = torch.from_numpy(y.T).to(self.device)
        loss1 = torch.sum((y_pred - y_target)**2)/(2.0*y.shape[0])
        loss2 = (self.reg_coef/2.0)*(torch.sum(self.W1**2) + torch.sum(self.W2**2))
        real_loss = loss1 + loss2

        if engagement_coef is None:
            real_loss.backward()
            self.update_rule()
        else:
            loss_per_task = torch.zeros(size=(len(engagement_coef),)).to(self.device)
            for i, phi in enumerate(engagement_coef):
                output_range = self.task_output_index[i]
                y_target_i = y[:, output_range[0]:output_range[1]]
                y_target_i = torch.from_numpy(y_target_i.T).to(self.device)
                y_pred_i = y_pred[output_range[0]:output_range[1], :]
                loss_i = phi*torch.sum((y_pred_i - y_target_i)**2)/(2.0*y.shape[0])
                loss_per_task[i] = loss_i
            # loss_per_task = torch.Tensor(loss_per_task)
            loss2 = (self.reg_coef / 2.0) * (torch.sum(self.W1 ** 2) + torch.sum(self.W2 ** 2))
            loss = torch.sum(loss_per_task, dim=0) + loss2
            loss.backward()
            self.update_rule()

        return real_loss