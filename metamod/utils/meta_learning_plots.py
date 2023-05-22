import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from metamod.utils import ResultsManager

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.palettes import Viridis, Category10, Category20
from bokeh.io import export_svg
from scipy.special import softmax
from matplotlib.ticker import FormatStrFormatter


def plot_maml_results(ax, maml_pre_processed, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    min_variable_label = plot_kwargs["min_variable_label"]
    max_variable_label = plot_kwargs["max_variable_label"]
    min_variable = plot_kwargs["min_variable"]
    max_variable = plot_kwargs["max_variable"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]
    ax = ax.flatten()

    # MAML DYNAMICS CURVES
    per_task_loss = maml_pre_processed["per_task_loss"]
    colors = cm.viridis(np.linspace(0, 1, per_task_loss.shape[0]))
    steps = maml_pre_processed["steps"]
    for j, step in enumerate(steps):
        if j == 0:
            ax[1].plot(np.arange(per_task_loss.shape[-1]), np.mean(per_task_loss, axis=1)[j, :], color=colors[j],
                       lw=line_width, alpha=1)
        elif j == len(steps) - 1:
            ax[1].plot(np.arange(per_task_loss.shape[-1]), np.mean(per_task_loss, axis=1)[j, :], color=colors[j],
                       lw=line_width, alpha=1)#, label=variable_label+": "+str(max_variable))
        else:
            ax[1].plot(np.arange(per_task_loss.shape[-1]), np.mean(per_task_loss, axis=1)[j, :], color=colors[j],
                       lw=line_width, alpha=1)
        ax[1].set_ylim([5.8 * 10 ** -2, 0.075])
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set_xlabel("Eval training steps", fontsize=fontsize)
    ax[1].set_ylabel("Loss dynamics", fontsize=fontsize)
    ax[1].text(-0.15, 1.05, subplot_labels[1], transform=ax[1].transAxes,
                  size=fontsize, weight='bold')


    # MAML LOSS
    ax[0].scatter(steps[0], np.mean(per_task_loss, axis=1)[0, 1], color=colors[0,], label=min_variable_label)
    ax[0].scatter(steps[-1], np.mean(per_task_loss, axis=1)[-1, 1], color=colors[-1,], label=max_variable_label)
    ax[0].scatter(steps, np.mean(per_task_loss, axis=1)[:, 1], color=colors[np.arange(len(steps))])
    ax[0].set_xlabel("Optimized steps", fontsize=fontsize)
    ax[0].set_ylabel("MAML loss", fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[0].spines[['right', 'top']].set_visible(False)
    if "weight_legend_pos" in plot_kwargs.keys():
        ax[0].legend(fontsize=fontsize - 5, frameon=False,
                              loc="upper left",
                              bbox_to_anchor=plot_kwargs["weight_legend_pos"],
                              handletextpad=plot_kwargs["handletextpad"])
    else:
        ax[0].legend(fontsize=fontsize - 4, frameon=False)
    ax[0].text(-0.15, 1.1, subplot_labels[0], transform=ax[0].transAxes,
                  size=fontsize, weight='bold')
    return ax


def plot_optimal_lr(ax, spec_path_list, var_sweep, var_label, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]
    y_label = plot_kwargs["y_label"]

    optimal_lr = []
    for i, spec_path in enumerate(spec_path_list):
        results = ResultsManager(spec_path, verbose=False)
        if i == 0:
            base_lr = results.params["model_params"]["learning_rate"]
        opt_lr = results.results["control_signal"].detach().cpu().numpy()
        optimal_lr.append((opt_lr + 1)*base_lr)

    colors = cm.viridis(np.linspace(0, 1, len(var_sweep)))
    for i, lr in enumerate(optimal_lr):
        if i == 0 or i == len(optimal_lr) - 1:
            ax.plot(lr, color=colors[i], lw=line_width, label=var_label+"="+str(var_sweep[i]))
        else:
            ax.plot(lr, color=colors[i], lw=line_width)

    ax.set_xlabel("Task time", fontsize=fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize)
    ax.legend(fontsize=fontsize - 5, frameon=False)
    ax.text(-0.15, 1.1, subplot_labels, transform=ax.transAxes,
                  size=fontsize, weight='bold')
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.spines[['right', 'top']].set_visible(False)

    return ax