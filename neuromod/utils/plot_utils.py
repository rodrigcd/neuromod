import numpy as np

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.palettes import Viridis, Category10, Category20
from bokeh.io import export_svg


def plot_weight_ev(flat_W_t, flat_eq_W_t, sim_iters, eq_iters, title=""):
    weight_plot = figure(x_axis_label="iters", y_axis_label="Weights", title=title)
    for i in range(np.min([flat_W_t.shape[-1], 20])):
        if i == 0:
            weight_plot.line(sim_iters, flat_W_t[:, i], line_width=6, line_dash=(4, 4), alpha=0.5, color=Category20[20][i],
                             legend_label="Simulation")
            weight_plot.line(eq_iters, flat_eq_W_t[:, i], line_width=3, color=Category20[20][i],
                             legend_label="First order")
        else:
            weight_plot.line(sim_iters, flat_W_t[:, i], line_width=6, line_dash=(4, 4), alpha=0.5, color=Category20[20][i])
            weight_plot.line(eq_iters, flat_eq_W_t[:, i], line_width=3, color=Category20[20][i])
    weight_plot.legend.location = "bottom_right"
    # weight_plot.output_backend = "svg"
    return weight_plot


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