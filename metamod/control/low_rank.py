from metamod.control import LinearNetControl
import torch


class LinearNetLowRankG(LinearNetControl):

    def __init__(self, in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                 intrinsic_noise, learning_rate=1e-5, n_steps=10000, time_constant=1.0,
                 control_lower_bound=0.0, control_upper_bound=0.5, init_g=None, gamma=0.99, cost_coef=0.3,
                 reward_convertion=1.0, control_lr=1e-4, control_rank="full"):

        super().__init__(in_out_cov, in_cov, out_cov, init_weights, reg_coef,
                         intrinsic_noise, learning_rate, n_steps, time_constant,
                         control_lower_bound, control_upper_bound, init_g, gamma, cost_coef,
                         reward_convertion, control_lr)

        # TODO: how to represent the control signal as low rank ...