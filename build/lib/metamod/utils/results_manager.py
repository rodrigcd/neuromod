import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.palettes import Viridis, Category10, Category20
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
output_notebook()

from metamod.utils import plot_lines, plot_weight_ev, load_results, plot_control, plot_net_reward


class ResultsManager(object):

    def __init__(self, resuts_path, verbose=False):
        self.results_path = resuts_path
        self.params, self.results = load_results(self.results_path)
        self.results["W1_sim"], self.results["W2_sim"] = self.results["weights_sim"]
        self._plot_settings()

        if verbose:
            print("### Results from", self.results_path, "###")
            print("### Param Keys ###")
            for key in self.params.keys():
                print("---", key, "---")
                mssg = []
                for sub_key in self.params[key].keys():
                    mssg.append(sub_key)
                print(", ".join(mssg))
            print("### Results Keys ###")
            print(self.results.keys())

    def _plot_settings(self):
        # Plot loss
        self.plot_loss_settings = {"losses_labels": ("Loss_t_sim", "Loss_t_eq", "Loss_t_control_opt", "Loss_t_sim_OPT"),
                                   "colors": (Category10[10][0], Category10[10][0], Category10[10][1], Category10[10][1]),
                                   "labels": ("Simulation", "Equation", "Optimized equation with control", "Optimized simulation"),
                                   "alphas": (0.3, 1, 1, 0.3),
                                   "plot_size": (800, 500),
                                   "x_axis_label": "iters",
                                   "y_axis_label": "Loss",
                                   "line_width": 3}

        self.plot_weights_settings = {"weights_label1": ("W1_sim", "W2_sim"),
                                      "weights_label2": ("W1_t_eq", "W2_t_eq"),
                                      "iters_set1_label": "weights_iters_sim",
                                      "iters_set2_label": "iters",
                                      "legends": ("Simulation", "First order"),
                                      "titles": ("W1", "W2"),
                                      "plot_size": (600, 500)}

        self.plot_control_settings = {"plot_size": (600, 500)}

        self.plot_net_reward_settings = {"legends": ("Baseline", "Training with control"),
                                         "losses_labels": ("Loss_t_sim", "Loss_t_sim_OPT")}

    def plot_loss(self, plot_settings=None):
        iters = self.results["iters"]
        if plot_settings is None:
            plot_settings = self.plot_loss_settings
        losses = ()
        for l_label in plot_settings["losses_labels"]:
            losses += (self.results[l_label],)
        aux_dict = plot_settings.copy()
        aux_dict.pop("losses_labels")
        s = plot_lines(iters, losses, **aux_dict)
        show(s)
        # return iters, losses, plot_settings

    def plot_weights(self, plot_settings=None):
        if plot_settings is None:
            plot_settings = self.plot_weights_settings

        W1_set1 = self.results[plot_settings["weights_label1"][0]]
        W2_set1 = self.results[plot_settings["weights_label1"][1]]
        W1_set2 = self.results[plot_settings["weights_label2"][0]]
        W2_set2 = self.results[plot_settings["weights_label2"][1]]

        weights_iter = self.results[plot_settings["iters_set1_label"]]
        iters = self.results[plot_settings["iters_set2_label"]]

        flat_W1_t = np.reshape(W1_set1, (W1_set1.shape[0], -1))
        flat_eq_W1_t = np.reshape(W1_set2, (W1_set2.shape[0], -1))

        flat_W2_t = np.reshape(W2_set1, (W2_set1.shape[0], -1))
        flat_eq_W2_t = np.reshape(W2_set2, (W2_set2.shape[0], -1))

        weight_plot1 = plot_weight_ev(flat_W1_t, flat_eq_W1_t, iters=weights_iter, iters_set2=iters,
                                      title=plot_settings["titles"][0], legend=plot_settings["legends"])
        weight_plot2 = plot_weight_ev(flat_W2_t, flat_eq_W2_t, iters=weights_iter, iters_set2=iters,
                                      title=plot_settings["titles"][1], legend=plot_settings["legends"])
        plot_size = plot_settings["plot_size"]
        grid = gridplot([weight_plot1, weight_plot2], ncols=2, width=plot_size[0], height=plot_size[1])
        show(grid)

    def plot_control_optimization(self):
        iter_control = self.params["control_params"]["iters_control"]
        cumulated_reward = self.results["cumulated_reward_opt"]
        opt = plot_lines(np.arange(iter_control), (cumulated_reward,), x_axis_label="gradient steps on control",
                         y_axis_label="Cumulated reward")
        show(opt)

    def plot_control_signal(self, plot_settings=None):
        if plot_settings is None:
            plot_settings = self.plot_control_settings
        G1_tilda, G2_tilda = self.results["control_signal"]
        G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
        G1_t = G1_tilda - np.ones(G1_tilda.shape)
        G2_t = G2_tilda - np.ones(G2_tilda.shape)

        flat_G1_t = np.reshape(G1_t, (G1_t.shape[0], -1))
        flat_G2_t = np.reshape(G2_t, (G2_t.shape[0], -1))
        iters = self.results["iters"]

        control_1 = plot_control(flat_G1_t, eq_iters=iters, title="G1")
        control_2 = plot_control(flat_G2_t, eq_iters=iters, title="G2")
        plot_size = plot_settings["plot_size"]
        grid = gridplot([control_1, control_2], ncols=2, width=plot_size[0], height=plot_size[1])
        show(grid)

    def plot_net_reward(self, plot_settings=None):
        if plot_settings is None:
            plot_settings = self.plot_net_reward_settings
        cost_coef = self.params["control_params"]["cost_coef"]
        reward_convertion = self.params["control_params"]["reward_convertion"]
        iters = self.results["iters"]

        G1_tilda, G2_tilda = self.results["control_signal"]
        G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
        G1_t = G1_tilda - np.ones(G1_tilda.shape)
        G2_t = G2_tilda - np.ones(G2_tilda.shape)

        control_cost = np.exp(cost_coef * (np.sum(G1_t ** 2, axis=(-1, -2)) + np.sum(G2_t ** 2, axis=(-1, -2)))) - 1
        baseline_control_cost = np.zeros(control_cost.shape)

        colors = (Category10[10][0], Category10[10][1])
        losses = ()
        for l_label in plot_settings["losses_labels"]:
            losses += (self.results[l_label],)
        control_costs = (baseline_control_cost, control_cost)
        legends = plot_settings["legends"]

        s, net_rewards, offset = plot_net_reward(iters, losses, control_costs, reward_convertion, labels=legends,
                                                 colors=colors)
        show(s)