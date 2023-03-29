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
from .plot_utils import compute_all_weight_sparsity

# plt.rcParams['text.usetex'] = True


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
    # ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
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


def task_switch_weights_plot(result_manager, **plot_kwargs):
    figsize = plot_kwargs["figsize"]
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]
    n_weights = plot_kwargs["n_weights"]
    xlim = plot_kwargs["xlim"]

    f, ax = plt.subplots(2, 2, figsize=figsize)
    ax = ax.flatten()
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
    #ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_ylabel(r"$W_{1}(t)$", fontsize=fontsize)
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
    #ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_ylabel(r"$W_{2}(t)$", fontsize=fontsize)
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
    ax[plot_index].set_ylabel(r"$G_{1}(t)$", fontsize=fontsize)
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
    ax[plot_index].set_ylabel(r"$G_{2}(t)$", fontsize=fontsize)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlim(xlim)


def non_linear_network_plots(result_manager, **plot_kwargs):
    figsize = plot_kwargs["figsize"]
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]
    n_weights = plot_kwargs["n_weights"]
    xlim = plot_kwargs["xlim"]
    xlim2 = plot_kwargs["xlim2"]
    alpha = plot_kwargs["alpha"]
    window_size = plot_kwargs["window_size"]

    f, ax = plt.subplots(3, 4, figsize=figsize)
    f.delaxes(ax[2, 0])
    f.delaxes(ax[2, -1])
    ax = ax.flatten()

    ### LOSS PLOT ###
    plot_index = 0
    loss_t_base = result_manager.results["Loss_t_eq"]
    loss_t_control = result_manager.results["Loss_t_control_opt"]
    loss_sim_control = result_manager.results["Loss_t_sim_OPT"]
    loss_sim_base = result_manager.results["Loss_t_sim"]
    iters = result_manager.results["iters"]
    ax[plot_index].plot(iters, loss_sim_base, "k", lw=line_width, label="Sim baseline", alpha=alpha)
    ax[plot_index].plot(iters, loss_sim_control, "C0", lw=line_width, label="Sim Control", alpha=alpha)
    ax[plot_index].plot(iters, loss_t_base, 'k', lw=line_width, label="Baseline")
    ax[plot_index].plot(iters, loss_t_control, 'C0', lw=line_width, label="Control")
    ax[plot_index].legend(fontsize=fontsize-4, frameon=False)
    #ax[0].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_xlim(xlim)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$\mathcal{L}(t)$", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[0], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_yscale("log")

    ### REWARD PLOTS ###
    plot_index = 1
    loss_t_base = result_manager.results["Loss_t_eq"]
    loss_t_control = result_manager.results["Loss_t_control_opt"]
    loss_sim_control = result_manager.results["Loss_t_sim_OPT"]
    loss_sim_base = result_manager.results["Loss_t_sim"]
    iters = result_manager.results["iters"]
    baseline_iRR = -result_manager.params["control_params"]["reward_convertion"] * loss_t_base
    control_iRR = -result_manager.params["control_params"]["reward_convertion"] * loss_t_control\
                          - result_manager.params["control_params"]["control"].control_cost(get_numpy=True)
    base_sim_iRR = -result_manager.params["control_params"]["reward_convertion"] * loss_sim_base
    control_sim_iRR = -result_manager.params["control_params"]["reward_convertion"] * loss_sim_control\
                          - result_manager.params["control_params"]["control"].control_cost(get_numpy=True)
    C = 0#-np.amin(control_iRR)
    ax[plot_index].plot(iters, base_sim_iRR+C, "k", lw=line_width, alpha=alpha)
    ax[plot_index].plot(iters, control_sim_iRR+C, "C0", lw=line_width, alpha=alpha)
    ax[plot_index].plot(iters, baseline_iRR+C, 'k', lw=line_width)
    ax[plot_index].plot(iters, control_iRR+C, 'C0', lw=line_width)
    # ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    # ax[0].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_xlim(xlim)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$v(t)$", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    # ax[plot_index].set_yscale("log")
    ax[plot_index].set_xlim(xlim2)

    ### GET WEIGHT MEASUREMENTS ###
    W1_t, W2_t = result_manager.results["W1_t_eq"], result_manager.results["W2_t_eq"]
    W1_control, W2_control = result_manager.results["W1_t_control_opt"], result_manager.results["W2_t_control_opt"]
    sim_iters = result_manager.results["weights_iters_sim"]
    W1_sim_base, W2_sim_base = result_manager.results["weights_sim"]
    W1_sim_control, W2_sim_control = result_manager.results["weights_sim_OPT"]
    base_l1, base_l2 = compute_all_weight_sparsity([W1_t, W2_t])
    con_l1, con_l2 = compute_all_weight_sparsity([W1_control, W2_control])
    sim_base_l1, sim_base_l2 = compute_all_weight_sparsity([W1_sim_base, W2_sim_base])
    sim_con_l1, sim_con_l2 = compute_all_weight_sparsity([W1_sim_control, W2_sim_control])

    ### W1 WEIGHTS ###
    plot_index = 2
    wt_base_eq = weight_flatten(W1_t)
    wt_base_sim = weight_flatten(W1_sim_base)
    weight1_ids = np.random.choice(np.arange(wt_base_sim.shape[-1]), replace=False, size=n_weights).astype(int)
    for weight_id in range(n_weights):
        if weight_id == 0:
            ax[plot_index].plot(iters, wt_base_eq[:, weight1_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width, label="Approx")
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight1_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width, label="Simulation")
        else:
            ax[plot_index].plot(iters, wt_base_eq[:, weight1_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width)
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight1_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width)
    ax[plot_index].legend(fontsize=fontsize-4, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$W_{1}(t)$", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                        size=fontsize, weight='bold')
    # ax[plot_index].set_yscale("log")
    ax[plot_index].set_xlim(xlim)

    ### W2 WEIGHTS ###
    plot_index = 3
    wt_base_eq = weight_flatten(W2_t)
    wt_base_sim = weight_flatten(W2_sim_base)
    weight2_ids = np.random.choice(np.arange(wt_base_sim.shape[-1]), replace=False, size=n_weights).astype(int)
    for weight_id in range(n_weights):
        if weight_id == 0:
            ax[plot_index].plot(iters, wt_base_eq[:, weight2_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width, label="Approx")
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight2_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width, label="Simulation")
        else:
            ax[plot_index].plot(iters, wt_base_eq[:, weight2_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width)
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight2_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width)
    # ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$W_{2}(t)$", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                        size=fontsize, weight='bold')
    # ax[plot_index].set_yscale("log")
    ax[plot_index].set_xlim(xlim)

    ### W1 CONTROL WEIGHTS ###
    plot_index = 6
    wt_base_eq = weight_flatten(W1_control)
    wt_base_sim = weight_flatten(W1_sim_control)
    # weight1_ids = np.random.choice(np.arange(wt_base_sim.shape[-1]), replace=False, size=n_weights).astype(int)
    for weight_id in range(n_weights):
        if weight_id == 0:
            ax[plot_index].plot(iters, wt_base_eq[:, weight1_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width, label="Approx")
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight1_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width, label="Simulation")
        else:
            ax[plot_index].plot(iters, wt_base_eq[:, weight1_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width)
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight1_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width)
    # ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$W_{1}(t)$", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15 , 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                        size=fontsize, weight='bold')
    # ax[plot_index].set_yscale("log")
    ax[plot_index].set_xlim(xlim)

    ### W2 CONTROL WEIGHTS ###
    plot_index = 7
    wt_base_eq = weight_flatten(W2_control)
    wt_base_sim = weight_flatten(W2_sim_control)
    # weight1_ids = np.random.choice(np.arange(wt_base_sim.shape[-1]), replace=False, size=n_weights).astype(int)
    for weight_id in range(n_weights):
        if weight_id == 0:
            ax[plot_index].plot(iters, wt_base_eq[:, weight2_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width, label="Approx")
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight2_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width, label="Simulation")
        else:
            ax[plot_index].plot(iters, wt_base_eq[:, weight2_ids[weight_id]],
                                "C"+str(weight_id)+"--", lw=line_width)
            ax[plot_index].plot(sim_iters, wt_base_sim[:, weight2_ids[weight_id]],
                                "C" + str(weight_id), lw=line_width)
    #ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$W_{2}(t)$", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15 , 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                        size=fontsize, weight='bold')
    # ax[plot_index].set_yscale("log")
    ax[plot_index].set_xlim(xlim)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)

    ### L1 LOSS ###
    plot_index = 4
    ax[plot_index].plot(sim_iters, sim_base_l1, "k--", lw=line_width, label="Sim baseline")
    ax[plot_index].plot(sim_iters, sim_con_l1, "C0--", lw=line_width, label="Sim control")
    ax[plot_index].plot(iters, base_l1, 'k', lw=line_width, label="Approx baseline")
    ax[plot_index].plot(iters, con_l1, 'C0', lw=line_width, label="Approx control")
    ax[plot_index].legend(fontsize=fontsize - 4, frameon=False)
    ax[plot_index].set_xlim(xlim)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$L_{1}$ norm", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)

    ### L2 LOSS ###
    plot_index = 5
    ax[plot_index].plot(sim_iters, sim_base_l2, "k--", lw=line_width)
    ax[plot_index].plot(sim_iters, sim_con_l2, "C0--", lw=line_width)
    ax[plot_index].plot(iters, base_l2, 'k', lw=line_width)
    ax[plot_index].plot(iters, con_l2, 'C0', lw=line_width)
    ax[plot_index].set_xlim(xlim)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[plot_index].set_ylabel("$L_{2}$ norm", fontsize=fontsize)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')

    ### G1 CONTROL ###
    plot_index = 9
    G = result_manager.params["control_params"]["control"].g1.detach().cpu().numpy()
    g_flatten = weight_flatten(G)
    g_flatten = np.stack([sliding_window(g_flatten[:, i], half_window=window_size) for i in range(g_flatten.shape[1])],
                         axis=1)
    iters = result_manager.results["iters"]
    for weight_id in range(n_weights):
        ax[plot_index].plot(iters, g_flatten[:, weight1_ids[weight_id]], lw=line_width, color="C"+str(weight_id))
    #ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_ylabel(r"$G_{1}(t)$", fontsize=fontsize)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlim(xlim)

    ### G2 CONTROL ###
    plot_index = 10
    G = result_manager.params["control_params"]["control"].g2.detach().cpu().numpy()
    g_flatten = weight_flatten(G)
    g_flatten = np.stack([sliding_window(g_flatten[:, i], half_window=window_size) for i in range(g_flatten.shape[1])],
                         axis=1)
    iters = result_manager.results["iters"]
    for weight_id in range(n_weights):
        ax[plot_index].plot(iters, g_flatten[:, weight2_ids[weight_id]], lw=line_width, color="C"+str(weight_id))
    #ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_ylabel(r"$G_{2}(t)$", fontsize=fontsize)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    ax[plot_index].set_xlim(xlim)

    return f, ax

def weight_flatten(W):
    return np.reshape(W, newshape=(W.shape[0], -1))


def sliding_window(vals, half_window=10):
    smooth_vals = []
    for i in range(len(vals)):
        min_index = np.max([i-half_window, 0])
        max_index = np.min([i+half_window, len(vals)])
        smooth_vals.append(np.mean(vals[min_index:max_index]))
    return np.array(smooth_vals)