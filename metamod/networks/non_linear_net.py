import torch
from metamod.networks import LinearNet


class NonLinearNet(LinearNet):

    def hidden_layer(self, x_tensor, g1_tilda=None):
        if g1_tilda is None:
            linear_hidden = self.W1 @ x_tensor
        else:
            linear_hidden = (self.W1 * g1_tilda) @ x_tensor
        non_linear_hidden = torch.tanh(linear_hidden)
        return non_linear_hidden, linear_hidden

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