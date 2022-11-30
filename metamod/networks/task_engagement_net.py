from metamod.networks import LinearNet
import torch


class LinearTaskEngNetwork(LinearNet):

    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=1e-5, reg_coef=0.1, W1_0=None, W2_0=None,
                 intrinsic_noise=1.0, task_output_index=()):
        super().__init__(input_dim, output_dim, hidden_dim, learning_rate, reg_coef, W1_0, W2_0,
                         intrinsic_noise)
        self.task_output_index = task_output_index

    def train_step(self, x, y, g1_tilda=None, g2_tilda=None, engagement_coeff=None):
        # Engagement coefficient should be equal to the number of tasks
        if g1_tilda is None:
            y_pred = self.forward(x)
        else:
            y_pred = self.controlled_forward(x, g1_tilda=g1_tilda, g2_tilda=g2_tilda)
        if engagement_coeff is None:
            y_target = torch.from_numpy(y.T).to(self.device)
            loss1 = torch.sum((y_pred - y_target)**2)/(2.0*y.shape[0])
            loss2 = (self.reg_coef / 2.0) * (torch.sum(self.W1 ** 2) + torch.sum(self.W2 ** 2))
            loss = loss1 + loss2
            loss.backward()
            self.update_rule()
            return loss
        else:
            loss_per_task = []
            for i, phi in engagement_coeff:
                output_range = self.task_output_index[i]
                y_target_i = y[:, output_range[0]:output_range[1]]
                y_target_i = torch.from_numpy(y_target_i).to(self.device)
                y_pred_i = y[:, output_range[0]:output_range[1]]
                loss_i = torch.sum((y_pred_i - y_target_i)**2)/(2.0*y.shape[0])
                loss_per_task.append(loss_i)
            loss_per_task = torch.cat(loss_per_task, dim=0)
            loss2 = (self.reg_coef / 2.0) * (torch.sum(self.W1 ** 2) + torch.sum(self.W2 ** 2))
            loss = torch.sum(loss_per_task, dim=0) + loss2
            loss.backwards()
            self.update_rule()
            return loss