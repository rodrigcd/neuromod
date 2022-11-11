import numpy as np

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.palettes import Viridis, Category10, Category20
from bokeh.io import export_svg


def plot_weight_ev(flat_W_t, flat_W_t_set2, iters, iters_set2, title="W", legend=("Simulation", "Equation")):
    weight_plot = figure(x_axis_label="iters", y_axis_label="Weights", title=title)
    for i in range(np.min([flat_W_t.shape[-1], 20])):
        if i == 0:
            weight_plot.line(iters, flat_W_t[:, i], line_width=6, line_dash=(4, 4), alpha=0.5, color=Category20[20][i],
                             legend_label=legend[0])
            weight_plot.line(iters_set2, flat_W_t_set2[:, i], line_width=3, color=Category20[20][i],
                             legend_label=legend[1])
        else:
            weight_plot.line(iters, flat_W_t[:, i], line_width=6, line_dash=(4, 4), alpha=0.5, color=Category20[20][i])
            weight_plot.line(iters_set2, flat_W_t_set2[:, i], line_width=3, color=Category20[20][i])
    weight_plot.legend.location = "bottom_right"
    # weight_plot.output_backend = "svg"
    return weight_plot


def plot_control(flat_G_t, eq_iters, title=""):
    control_plot = figure(x_axis_label="iters", y_axis_label="G(t) modulation", title=title)
    for i in range(np.min([flat_G_t.shape[-1], 20])):
        control_plot.line(eq_iters, flat_G_t[:, i], line_width=3, color=Category20[20][i])
    # weight_plot.output_backend = "svg"
    return control_plot


def plot_lines(x_vals, y_val_list=(), labels=(), alphas=(), colors=(), plot_size=(800, 500), x_axis_label="iters",
               y_axis_label="Loss", line_width=3):
    s = figure(x_axis_label=x_axis_label, y_axis_label=y_axis_label, width=plot_size[0], height=plot_size[1])
    for i, loss in enumerate(y_val_list):
        if len(alphas) == 0:
            alpha = 1.0
        else:
            alpha = alphas[i]

        if len(colors) == 0:
            color = Category20[20][i]
        else:
            color = colors[i]

        if len(labels) == 0:
            label = ""
        else:
            label = labels[i]

        s.line(x_vals, loss, line_width=line_width, alpha=alpha, legend_label=label, color=color)
    return s


def plot_net_reward(x_vals, losses=(), control_costs=(), reward_convertion=1.0, labels=(), alphas=(), colors=(),
                    y_axis_label=("Net reward (+offset)", "Cumulated net reward"), plot_size=(800, 500)):
        iRR = [-reward_convertion*loss for loss in losses]

        net_rewards = [iRR[i]-control_costs[i] for i in range(len(losses))]
        net_r_mins = [np.amin(net_rewards[i]) for i in range(len(losses))]
        offset = np.amin(net_r_mins)
        s = plot_lines(x_vals=x_vals, y_val_list=net_rewards-offset, labels=labels, alphas=alphas, colors=colors,
                       y_axis_label=y_axis_label[0])
        s.legend.location = "bottom_right"

        cumulated_net_reward = [np.cumsum(net_rewards[i]-offset) for i in range(len(losses))]
        s2 = plot_lines(x_vals=x_vals, y_val_list=cumulated_net_reward, labels=labels, alphas=alphas, colors=colors,
                        y_axis_label=y_axis_label[1])
        s2.legend.location = "bottom_right"

        grid = gridplot([s, s2], ncols=2, width=plot_size[0], height=plot_size[1])
        return grid, net_rewards, offset