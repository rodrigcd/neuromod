import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.palettes import Viridis, Category10, Category20
from bokeh.io import export_svg
np.random.seed(0)  # For picking random weights

plt.rcParams['text.usetex'] = True


def two_layer_parameters_plot(result_manager, **plot_kwargs):
    figsize = plot_kwargs["figsize"]
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]
    n_weights = plot_kwargs["n_weights"]
    xlim = plot_kwargs["xlim"]

    f, ax = plt.subplots(1, 4, figsize=figsize)

    ## W1 WEIGHT PL0T ##
    plot_index = 0
    w_t_baseline = weight_flatten(result_manager.results["W1_t_eq"])
    w_t_control = weight_flatten(result_manager.results["W1_t_control_opt"])
    weight1_ids = np.random.choice(np.arange(w_t_baseline.shape[-1]), replace=False, size=n_weights).astype(int)
    iters = result_manager.results["iters"]
    for weight_id in range(n_weights):
        if weight_id == 0:
            ax[plot_index].plot(iters, w_t_baseline[:, weight1_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width, label="Baseline")
            ax[plot_index].plot(iters, w_t_control[:, weight1_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width, label="Control")
        else:
            ax[plot_index].plot(iters, w_t_baseline[:, weight1_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width)
            ax[plot_index].plot(iters, w_t_control[:, weight1_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width)
    ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_title(r"$W_{1}(t)$", fontsize=fontsize)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlim(xlim)

    ## W2 WEIGHT PL0T ##
    plot_index = 1
    w_t_baseline = weight_flatten(result_manager.results["W2_t_eq"])
    w_t_control = weight_flatten(result_manager.results["W2_t_control_opt"])
    weight2_ids = np.random.choice(np.arange(w_t_baseline.shape[-1]), replace=False, size=n_weights).astype(int)
    iters = result_manager.results["iters"]
    for weight_id in range(n_weights):
        if weight_id == 0:
            ax[plot_index].plot(iters, w_t_baseline[:, weight2_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width, label="Baseline")
            ax[plot_index].plot(iters, w_t_control[:, weight2_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width, label="Control")
        else:
            ax[plot_index].plot(iters, w_t_baseline[:, weight2_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width)
            ax[plot_index].plot(iters, w_t_control[:, weight2_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width)
    # ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_title(r"$W_{2}(t)$", fontsize=fontsize)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlim(xlim)

    ## G1 WEIGHT PL0T ##
    plot_index = 2
    G = result_manager.params["control_params"]["control"].g1.detach().cpu().numpy()
    g_flatten = weight_flatten(G)
    iters = result_manager.results["iters"]
    for weight_id in range(n_weights):
        ax[plot_index].plot(iters, g_flatten[:, weight1_ids[weight_id]], lw=line_width, color="C"+str(weight_id))
    #ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_title(r"$G_{1}(t)$", fontsize=fontsize)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlim(xlim)

    ## G2 WEIGHT PL0T ##
    plot_index = 3
    G = result_manager.params["control_params"]["control"].g2.detach().cpu().numpy()
    g_flatten = weight_flatten(G)
    iters = result_manager.results["iters"]
    for weight_id in range(n_weights):
        ax[plot_index].plot(iters, g_flatten[:, weight2_ids[weight_id]], lw=line_width, color="C"+str(weight_id))
    #ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_title(r"$G_{2}(t)$", fontsize=fontsize)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlim(xlim)

def weight_flatten(W):
    return np.reshape(W, newshape=(W.shape[0], -1))