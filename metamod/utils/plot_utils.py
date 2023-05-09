import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.palettes import Viridis, Category10, Category20
from bokeh.io import export_svg
from scipy.special import softmax

# plt.rcParams['text.usetex'] = True


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


def single_task_plot(manager_list, ax=None, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    skip_xlabel = plot_kwargs["skip_xlabel"]
    label_in_title = plot_kwargs["label_in_title"]
    subplot_labels = plot_kwargs["subplot_labels"]
    x_lim = None
    if "x_lim" in plot_kwargs.keys():
        x_lim = plot_kwargs["x_lim"]

    #############
    # Loss plot #
    #############
    plot_index = 1
    baseline_losses = []
    control_losses = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_losses.append(results.results["Loss_t_eq"])
        control_losses.append(results.results["Loss_t_control_opt"])

    mean_baseline_losses = np.mean(np.stack(baseline_losses, axis=0), axis=0)
    mean_control_losses = np.mean(np.stack(control_losses, axis=0), axis=0)
    std_baseline_losses = np.std(np.stack(baseline_losses, axis=0), axis=0)
    std_control_losses = np.std(np.stack(control_losses, axis=0), axis=0)

    ax[plot_index].plot(iters, mean_baseline_losses, 'k', lw=line_width, label="Baseline")
    ax[plot_index].plot(iters, mean_control_losses, 'C0', lw=line_width, label="Controlled")
    ax[plot_index].fill_between(iters, mean_baseline_losses - 2*std_baseline_losses, mean_baseline_losses + 2*std_baseline_losses,
                    color='k', alpha=0.3)
    ax[plot_index].fill_between(iters, mean_control_losses - 2*std_control_losses, mean_control_losses + 2*std_control_losses,
                    color='C0', alpha=0.3)
    if not skip_xlabel:
        ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_xlim(x_lim)
    if label_in_title:
        ax[plot_index].set_title(r"$\mathcal{L}(t)$", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel(r"$\mathcal{L}(t)$", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')

    # ax[plot_index].legend(fontsize=fontsize - 2)

    ###########################
    # Net instant reward rate #
    ###########################
    plot_index = 0
    baseline_r = []
    control_r = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        G1_tilda, G2_tilda = results.results["control_signal"]
        G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
        G1_t = G1_tilda - np.ones(G1_tilda.shape)
        G2_t = G2_tilda - np.ones(G2_tilda.shape)
        cost_coef = results.params["control_params"]["cost_coef"]
        reward_conv = results.params["control_params"]["reward_convertion"]
        control_cost = compute_control_cost(G1_t, G2_t, cost_coef=cost_coef)
        r_control = -reward_conv*results.results["Loss_t_control_opt"] - control_cost
        r_base = -reward_conv*results.results["Loss_t_eq"]
        baseline_r.append(r_base)
        control_r.append(r_control)

    mean_baseline_r = np.mean(np.stack(baseline_r, axis=0), axis=0)
    mean_control_r = np.mean(np.stack(control_r, axis=0), axis=0)
    std_baseline_r = np.std(np.stack(baseline_r, axis=0), axis=0)
    std_control_r = np.std(np.stack(control_r, axis=0), axis=0)

    ax[plot_index].plot(iters, mean_baseline_r, 'k', lw=line_width, label="Baseline")
    ax[plot_index].plot(iters, mean_control_r, 'C0', lw=line_width, label="Control")
    ax[plot_index].fill_between(iters, mean_baseline_r - 2*std_baseline_r, mean_baseline_r + 2*std_baseline_r,
                    color='k', alpha=0.3)
    ax[plot_index].fill_between(iters, mean_control_r - 2*std_control_r, mean_control_r + 2*std_control_r,
                    color='C0', alpha=0.3)
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    if not skip_xlabel:
        ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title(r"$v(t)$", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel(r"$v(t)$", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    # ax[plot_index].text(.01, .99, '(a)', ha='left', va='top', transform=ax[plot_index].transAxes)

    ###################
    # Weight sparsity #
    ###################
    plot_index = 2
    baseline_l1 = []
    baseline_l2 = []
    control_l1 = []
    control_l2 = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        W1_t, W2_t = results.results["W1_t_eq"], results.results["W2_t_eq"]
        W1_control, W2_control = results.results["W1_t_control_opt"], results.results["W2_t_control_opt"]
        base_l1, base_l2 = compute_all_weight_sparsity([W1_t, W2_t])
        con_l1, con_l2 = compute_all_weight_sparsity([W1_control, W2_control])
        baseline_l1.append(base_l1)
        baseline_l2.append(base_l2)
        control_l1.append(con_l1)
        control_l2.append(con_l2)

    baseline_l1 = np.mean(np.stack(baseline_l1, axis=0), axis=0)
    baseline_l2 = np.mean(np.stack(baseline_l2, axis=0), axis=0)
    control_l1 = np.mean(np.stack(control_l1, axis=0), axis=0)
    control_l2 = np.mean(np.stack(control_l2, axis=0), axis=0)
    ax[plot_index].plot(iters, baseline_l1, 'k', lw=line_width, label=r"Base $L1$")
    ax[plot_index].plot(iters, control_l1, 'C0', lw=line_width, label=r"Ctrl $L1$")
    ax[plot_index].plot(iters, baseline_l2, 'k--', lw=line_width, label=r"Base $L2$")
    ax[plot_index].plot(iters, control_l2, 'C0--', lw=line_width, label=r"Ctrl $L2$")
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-5, frameon=False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title("Weight norms", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("Weight norms", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')

    ################
    # Control Size #
    ################
    plot_index = 3
    normalize = True
    base_loss_der = []
    ctrl_loss_der = []
    G1_size = []
    G2_size = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        G1_tilda, G2_tilda = results.results["control_signal"]
        G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
        G1_t = G1_tilda - np.ones(G1_tilda.shape)
        G2_t = G2_tilda - np.ones(G2_tilda.shape)
        _, G1_norm = compute_weight_sparsity(G1_t)
        _, G2_norm = compute_weight_sparsity(G2_t)
        G1_size.append(G1_norm)
        G2_size.append(G2_norm)
        b_der_loss = np.diff(results.results["Loss_t_eq"])
        c_der_loss = np.diff(results.results["Loss_t_control_opt"])
        b_der_loss = np.concatenate([b_der_loss, (b_der_loss[-1],)])
        c_der_loss = np.concatenate([c_der_loss, (c_der_loss[-1],)])
        base_loss_der.append(b_der_loss)
        ctrl_loss_der.append(c_der_loss)

    base_loss_der = np.mean(np.stack(base_loss_der, axis=0), axis=0)
    ctrl_loss_der = np.mean(np.stack(ctrl_loss_der, axis=0), axis=0)
    G1_size = np.mean(np.stack(G1_size, axis=0), axis=0)
    G2_size = np.mean(np.stack(G2_size, axis=0), axis=0)
    if normalize:
        base_loss_der = -base_loss_der / np.max(np.abs(base_loss_der))
        ctrl_loss_der = -ctrl_loss_der / np.max(np.abs(ctrl_loss_der))
        G1_size = G1_size / np.max(np.abs(G1_size))
        G2_size = G2_size / np.max(np.abs(G2_size))
    ax[plot_index].plot(iters, base_loss_der, 'k', lw=line_width, label=r"Base $d\mathcal{L}/dt$")
    ax[plot_index].plot(iters, ctrl_loss_der, 'C0', lw=line_width, label=r"Ctrl $d\mathcal{L}/dt$")
    ax[plot_index].plot(iters, G1_size, 'C2--', lw=line_width-1, label=r"$G_1(t)$ size")
    ax[plot_index].plot(iters, G2_size, 'C3--', lw=line_width-1, label=r"$G_2(t)$ size")
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-5, frameon=False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title("Normalized unit", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("Normalized unit", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')

    return ax


def task_switch_plot(result_manager, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    figsize = plot_kwargs["figsize"]
    zoom_xlim = plot_kwargs["zoom_xlim"]
    zoom_ylim = plot_kwargs["zoom_ylim"]
    subplot_labels = plot_kwargs["subplot_labels"]
    hspace = plot_kwargs["hspace"]
    legend_pos = plot_kwargs["legend_pos"]
    legend_pos2 = plot_kwargs["legend_pos2"]
    line_pos_y = plot_kwargs["line_pos_y"]

    fig = plt.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(2, 2, wspace=0.25, hspace=hspace)  # 2x2 grid
    ax0 = fig.add_subplot(gs[0, :])  # full first row
    ax1 = fig.add_subplot(gs[1, 0])  # second row, first col
    ax2 = fig.add_subplot(gs[1, 1])  # second row, second col

    #############
    # Loss plot #
    #############
    loss_t_eq = result_manager.results["Loss_t_eq"]
    loss_t_control = result_manager.results["Loss_t_control_opt"]
    iters = result_manager.results["iters"]
    ax0.plot(iters, loss_t_eq, 'k', lw=line_width, label="Baseline")
    ax0.plot(iters, loss_t_control, 'C0', lw=line_width, label="Controlled")
    ax0.legend(fontsize=fontsize-2, frameon=False, bbox_to_anchor=legend_pos, loc='upper left', borderaxespad=0)
    ax0.set_xlabel("Task time", fontsize=fontsize)
    ax0.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    # Zoom square
    alpha = 0.8
    ax0.set_ylim([-0.1, 2.9])
    ax0.set_xlim([np.amin(iters), np.amax(iters)])
    ax0.plot([zoom_xlim[0], zoom_xlim[1]], [zoom_ylim[0], zoom_ylim[0]], "C3--", lw=line_width, alpha=alpha)
    ax0.plot([zoom_xlim[0], zoom_xlim[1]], [zoom_ylim[1], zoom_ylim[1]], "C3--", lw=line_width, alpha=alpha)
    ax0.plot([zoom_xlim[0], zoom_xlim[0]], [zoom_ylim[0], zoom_ylim[1]], "C3--", lw=line_width, alpha=alpha)
    ax0.plot([zoom_xlim[1], zoom_xlim[1]], [zoom_ylim[0], zoom_ylim[1]], "C3--", lw=line_width, alpha=alpha)
    ax0.plot([zoom_xlim[0], 12800], [zoom_ylim[0], -line_pos_y], "C3--", lw=line_width, alpha=alpha, zorder=0, clip_on=False)
    ax0.plot([zoom_xlim[1], np.amax(iters)], [zoom_ylim[0], -line_pos_y], "C3--", lw=line_width, alpha=alpha, zorder=0, clip_on=False)
    ax0.set_ylabel("$\mathcal{L}(t)$", fontsize=fontsize)
    ax0.spines[['right', 'top']].set_visible(False)
    ax0.text(-0.15/2, 1.05, subplot_labels[0], transform=ax0.transAxes,
                  size=fontsize, weight='bold')


    #############
    # Zoom plot #
    #############
    G1_tilda, G2_tilda = result_manager.results["control_signal"]
    switch_every = result_manager.params["dataset_params"]["change_tasks_every"]
    n_steps = result_manager.params["equation_params"]["n_steps"]
    cost_coef = result_manager.params["control_params"]["cost_coef"]
    peak_iters = np.arange(int(n_steps/switch_every))*switch_every

    G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
    G1_t = G1_tilda - np.ones(G1_tilda.shape)
    G2_t = G2_tilda - np.ones(G2_tilda.shape)
    _, G1_norm = compute_weight_sparsity(G1_t)
    _, G2_norm = compute_weight_sparsity(G2_t)

    G_cost = cost_coef*(G1_norm + G2_norm)
    G_cost = G_cost/np.amax(G_cost)

    ax2.plot(iters, loss_t_eq, 'k', lw=line_width)
    ax2.plot(iters, loss_t_control, 'C0', lw=line_width)
    ax2.plot(iters, G_cost, "C2--", lw=line_width, label="Normalized $C(G(t))$")
    ax2.legend(fontsize=fontsize-2, frameon=False, bbox_to_anchor=legend_pos2, loc='upper left', borderaxespad=0)
    ax2.set_xlabel("Task time", fontsize=fontsize)
    ax2.set_xlim(zoom_xlim)
    ax2.set_ylim(zoom_ylim)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.text(-0.15, 1.05, subplot_labels[2], transform=ax2.transAxes,
                  size=fontsize, weight='bold')

    ################################
    # Loss peaks and control peaks #
    ################################
    ax1.plot(iters[peak_iters], loss_t_eq[peak_iters], 'k-o', lw=line_width)#, label="Baseline $\mathcal{L}(t)$")
    ax1.plot(iters[peak_iters], loss_t_control[peak_iters], 'C0-o', lw=line_width)#, label="Control $\mathcal{L}(t)$")
    ax1.plot(iters[peak_iters], G_cost[peak_iters], 'C2-o', lw=line_width) #label="Normalized $C(G(t))$")
    # ax1.legend(fontsize=fontsize-2)
    ax1.set_xlabel("Task time", fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax1.set_ylabel("Peak values", fontsize=fontsize)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.text(-0.15, 1.05, subplot_labels[1], transform=ax1.transAxes,
                  size=fontsize, weight='bold')
    return fig


def cat_assimilation_plot(result_manager1_list, result_manager2_list, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    figsize = plot_kwargs["figsize"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]
    xlim1 = None
    if "xlim1" in plot_kwargs.keys():
        xlim1 = plot_kwargs["xlim1"]

    f, ax = plt.subplots(2, 2, figsize=figsize)

    ### LOSS PLOT ###
    loss_diff = []
    for i, results in enumerate(result_manager1_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_loss = results.results["Loss_t_eq"]
        control_loss = results.results["Loss_t_control_opt"]
        loss_diff.append(baseline_loss - control_loss)

    mean_loss_diff = np.mean(np.stack(loss_diff, axis=0), axis=0)
    std_loss_diff = np.std(np.stack(loss_diff, axis=0), axis=0)

    ax[0, 0].plot(iters, mean_loss_diff, 'C0', lw=line_width, label="Controlled")
    ax[0, 0].fill_between(iters, mean_loss_diff - 2*std_loss_diff,
                          mean_loss_diff + 2*std_loss_diff,
                          color='C0', alpha=0.3)

    #ax[0, 0].legend(fontsize=fontsize-2)
    # ax[0, 0].set_xlabel("Task time", fontsize=fontsize)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[0, 0].set_ylabel(r"$\mathcal{L}_{B}(t)-\mathcal{L}_{C}(t)$", fontsize=fontsize)
    ax[0, 0].set_title("Semantic", fontsize=fontsize)
    ax[0, 0].spines[['right', 'top']].set_visible(False)
    ax[0, 0].text(-0.15, 1.05, subplot_labels[0, 0], transform=ax[0, 0].transAxes,
                  size=fontsize, weight='bold')
    ax[0, 0].set_xlim(xlim1)

    loss_diff = []
    for i, results in enumerate(result_manager2_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_loss = results.results["Loss_t_eq"]
        control_loss = results.results["Loss_t_control_opt"]
        loss_diff.append(baseline_loss - control_loss)

    mean_loss_diff = np.mean(np.stack(loss_diff, axis=0), axis=0)
    std_loss_diff = np.std(np.stack(loss_diff, axis=0), axis=0)

    ax[0, 1].plot(iters, mean_loss_diff, 'C0', lw=line_width, label="Controlled")
    ax[0, 1].fill_between(iters, mean_loss_diff - 2 * std_loss_diff,
                          mean_loss_diff + 2 * std_loss_diff,
                          color='C0', alpha=0.3)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    # ax[0, 1].set_ylabel(r"$\mathcal{L}_{B}(t)-\mathcal{L}_{C}(t)$", fontsize=fontsize)
    ax[0, 1].set_title("MNIST", fontsize=fontsize)
    ax[0, 1].set_xlim([0, 7000])
    ax[0, 1].spines[['right', 'top']].set_visible(False)
    ax[0, 1].text(-0.15, 1.05, subplot_labels[0, 1], transform=ax[0, 1].transAxes,
                  size=fontsize, weight='bold')

    ### NUS PLOT WITH TREE ###
    result_manager1 = result_manager1_list[0]
    result_manager2 = result_manager2_list[0]
    _, nus = result_manager1.results["nus"]
    iters = result_manager1.results["iters"]
    nus = nus.detach().cpu().numpy()
    n_levels = result_manager1.params["dataset_params"]["dataset"].h_level
    leaves_per_level = 2**(np.arange(n_levels))
    level_colors = ["C" + str(i) for i in range(n_levels)]
    level_per_curve = []
    current_level = 0
    for n_leaves in leaves_per_level:
        for i in range(n_leaves):
            level_per_curve.append(current_level)
        current_level += 1
    for i in range(nus.shape[-1]):
        ax[1, 0].plot(iters, nus[:, 0, 0, i], lw=line_width, color=level_colors[level_per_curve[i]])
    ax[1, 0].set_xlabel("Task time", fontsize=fontsize)
    ax[1, 0].set_ylabel(r"$\nu_2^b(t)$", fontsize=fontsize)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[1, 0].spines[['right', 'top']].set_visible(False)
    ax[1, 0].text(-0.15, 1.05, subplot_labels[1, 0], transform=ax[1, 0].transAxes,
                  size=fontsize, weight='bold')
    ax[1, 0].set_xlim(xlim1)

    # Drawing hierarchy tree
    root_point = (7500, 0.1)
    height = 0.05
    width = 4500
    marker_size = 80
    draw_tree(ax[1, 0], n_levels, level_colors, root_point, height=height,
              width=width, marker_size=marker_size, line_width=line_width)

    ### MNIST NUS PLOT ###
    iters = result_manager2.results["iters"]
    nus_list = []

    for i, results in enumerate(result_manager2_list):
        if i == 0:
            iters = results.results["iters"]
        _, nus = results.results["nus"]
        nus_list.append(nus.detach().cpu().numpy())

    mean_nus = np.mean(np.stack(nus_list, axis=0), axis=0)
    std_nus = np.std(np.stack(nus_list, axis=0), axis=0)

    for i in range(nus.shape[-1])[:10]:
        ax[1, 1].plot(iters, mean_nus[:, 0, 0, i], lw=line_width, label="Digit "+str(i))
    # ax[1, 1].legend()
    ax[1, 1].set_xlabel("Task time", fontsize=fontsize)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[1, 1].set_xlim([0, 7000])
    ax[1, 1].set_ylim([0, 0.0018])
    ax[1, 1].spines[['right', 'top']].set_visible(False)
    ax[1, 1].text(-0.15, 1.05, subplot_labels[1, 1], transform=ax[1, 1].transAxes,
                  size=fontsize, weight='bold')
    ax[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    current_digit = 0
    for p1 in range(2):
        for p2 in range(5):
            ax[1, 1].text(0.8*(1+p1*0.1), 0.8*(1-p2*0.15), str(current_digit), transform=ax[1, 1].transAxes,
                          size=fontsize+3, weight='bold', color="C"+str(current_digit))
            current_digit += 1

    # ax[1, 1].set_ylabel("\nu_{2}^{b}", fontsize=fontsize)


def task_engagement_plot(result_manager1_list, result_manager2_list, result_manager3_list, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    figsize = plot_kwargs["figsize"]
    line_width = plot_kwargs["line_width"]
    ylim2 = plot_kwargs["ylim2"]
    ylim3 = plot_kwargs["ylim3"]
    subplot_labels = plot_kwargs["subplot_labels"]
    x_lim = None
    if "x_lim" in plot_kwargs.keys():
        x_lim = plot_kwargs["x_lim"]
    f, ax = plt.subplots(3, 2, figsize=figsize)

    ### LOSS PLOT
    baseline_losses = []
    control_losses = []
    for i, results in enumerate(result_manager1_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_losses.append(results.results["Loss_t_eq"])
        control_losses.append(results.results["Loss_t_control_opt"])

    mean_baseline_losses = np.mean(np.stack(baseline_losses, axis=0), axis=0)
    mean_control_losses = np.mean(np.stack(control_losses, axis=0), axis=0)
    std_baseline_losses = np.std(np.stack(baseline_losses, axis=0), axis=0)
    std_control_losses = np.std(np.stack(control_losses, axis=0), axis=0)

    ax[1, 0].plot(iters, mean_baseline_losses, 'k', lw=line_width, label="Baseline")
    ax[1, 0].plot(iters, mean_control_losses, 'C0', lw=line_width, label="Controlled")
    ax[1, 0].fill_between(iters, mean_baseline_losses - 2*std_baseline_losses, mean_baseline_losses + 2*std_baseline_losses,
                    color='k', alpha=0.3)
    ax[1, 0].fill_between(iters, mean_control_losses - 2*std_control_losses, mean_control_losses + 2*std_control_losses,
                    color='C0', alpha=0.3)

    # ax[1, 0].set_ylabel(r"$\mathcal{L}(t)$", fontsize=fontsize)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[1, 0].set_ylabel("Active", fontsize=fontsize)
    #ax[1, 0].legend(fontsize=fontsize - 2, frameon=False)
    ax[1, 0].spines[['right', 'top']].set_visible(False)
    ax[1, 0].text(-0.15, 1.05, subplot_labels[1, 0], transform=ax[1, 0].transAxes,
                  size=fontsize, weight='bold')
    # ax[1, 0].set_xlabel("Task time", fontsize=fontsize)

    baseline_losses = []
    control_losses = []
    for i, results in enumerate(result_manager2_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_losses.append(results.results["Loss_t_eq"])
        control_losses.append(results.results["Loss_t_control_opt"])

    mean_baseline_losses = np.mean(np.stack(baseline_losses, axis=0), axis=0)
    mean_control_losses = np.mean(np.stack(control_losses, axis=0), axis=0)
    std_baseline_losses = np.std(np.stack(baseline_losses, axis=0), axis=0)
    std_control_losses = np.std(np.stack(control_losses, axis=0), axis=0)

    ax[0, 0].plot(iters, mean_baseline_losses, 'k', lw=line_width, label="Baseline")
    ax[0, 0].plot(iters, mean_control_losses, 'C0', lw=line_width, label="Controlled")
    ax[0, 0].fill_between(iters, mean_baseline_losses - 2*std_baseline_losses, mean_baseline_losses + 2*std_baseline_losses,
                    color='k', alpha=0.3)
    ax[0, 0].fill_between(iters, mean_control_losses - 2*std_control_losses, mean_control_losses + 2*std_control_losses,
                    color='C0', alpha=0.3)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[0, 0].set_ylabel("Attentive", fontsize=fontsize)
    ax[0, 0].set_title(r"$\mathcal{L}(t)$", fontsize=fontsize)
    ax[0, 0].spines[['right', 'top']].set_visible(False)
    ax[0, 0].text(-0.15, 1.05, subplot_labels[0, 0], transform=ax[0, 0].transAxes,
                  size=fontsize, weight='bold')
    ax[0, 0].legend(fontsize=fontsize - 3, frameon=False)

    ### Engagement plot ###
    phis = []
    for i, results in enumerate(result_manager1_list):
        if i == 0:
            iters = results.results["iters"]
        phis.append(results.results["final_engagement_coef"].detach().cpu().numpy())
    mean_phis = np.mean(np.stack(phis, axis=0), axis=0)
    std_phis = np.std(np.stack(phis, axis=0), axis=0)
    colors = ["C"+str(i) for i in range(mean_phis.shape[-1])]
    for i in range(mean_phis.shape[-1]):
        mean_phi = mean_phis[:, i]
        std_phi = std_phis[:, i]
        color = colors[i]
        legend = "Digits: " + str(result_manager1_list[0].params["dataset_params"]["dataset_list_params"][i]["subset"])
        ax[1, 1].fill_between(iters, mean_phi - 2 * std_phi,
                              mean_phi + 2 * std_phi,
                              color=color, alpha=0.3)
        ax[1, 1].plot(iters, mean_phi, color, lw=line_width, label=legend)
    #ax[1, 1].set_ylabel(r"$\psi_{\tau}(t)$", fontsize=fontsize)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=fontsize-2)

    # ax[1, 0].legend(fontsize=fontsize - 2, frameon=False)
    ax[1, 1].spines[['right', 'top']].set_visible(False)
    ax[1, 1].text(-0.15, 1.05, subplot_labels[1, 1], transform=ax[1, 1].transAxes,
                  size=fontsize, weight='bold')

    phis = []
    for i, results in enumerate(result_manager2_list):
        if i == 0:
            iters = results.results["iters"]
        phis.append(results.results["final_engagement_coef"].detach().cpu().numpy())
    mean_phis = np.mean(np.stack(phis, axis=0), axis=0)
    std_phis = np.std(np.stack(phis, axis=0), axis=0)
    colors = ["C"+str(i) for i in range(mean_phis.shape[-1])]
    for i in range(mean_phis.shape[-1]):
        mean_phi = mean_phis[:, i]
        std_phi = std_phis[:, i]
        color = colors[i]
        legend = "Digits: " + str(result_manager1_list[0].params["dataset_params"]["dataset_list_params"][i]["subset"])
        ax[0, 1].fill_between(iters, mean_phi - 2 * std_phi,
                              mean_phi + 2 * std_phi,
                              color=color, alpha=0.3)
        ax[0, 1].plot(iters, mean_phi, color, lw=line_width, label=legend)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fontsize-2)
    #ax[0, 1].set_xlabel("Task time", fontsize=fontsize)
    ax[0, 1].legend(fontsize=fontsize - 3, frameon=False)
    ax[0, 1].set_ylim(ylim2)
    ax[0, 1].spines[['right', 'top']].set_visible(False)
    ax[0, 1].text(-0.15, 1.05, subplot_labels[0, 1], transform=ax[0, 1].transAxes,
                  size=fontsize, weight='bold')
    ax[0, 1].set_title(r"$\psi_{\tau}(t)$", fontsize=fontsize)

    phis = []
    for i, results in enumerate(result_manager3_list):
        if i == 0:
            iters = results.results["iters"]
        phis.append(results.results["final_engagement_coef"].detach().cpu().numpy())
    mean_phis = np.mean(np.stack(phis, axis=0), axis=0)
    std_phis = np.std(np.stack(phis, axis=0), axis=0)
    colors = ["C"+str(i) for i in range(mean_phis.shape[-1])]
    for i in range(mean_phis.shape[-1]):
        mean_phi = mean_phis[:, i]
        std_phi = std_phis[:, i]
        color = colors[i]
        legend = "Digits: " + str(result_manager1_list[0].params["dataset_params"]["dataset_list_params"][i]["subset"])
        ax[2, 1].fill_between(iters, mean_phi - 2 * std_phi,
                              mean_phi + 2 * std_phi,
                              color=color, alpha=0.3)
        ax[2, 1].plot(iters, mean_phi, color, lw=line_width, label=legend)
    ax[2, 1].tick_params(axis='both', which='major', labelsize=fontsize-2)
    # ax[0, 1].set_xlabel("Task time", fontsize=fontsize)
    # ax[2, 1].legend(fontsize=fontsize - 2, frameon=False)
    # ax[2, 1].set_ylim(ylim2)
    ax[2, 1].spines[['right', 'top']].set_visible(False)
    ax[2, 1].text(-0.15, 1.05, subplot_labels[2, 1], transform=ax[2, 1].transAxes,
                  size=fontsize, weight='bold')
    # ax[2, 1].set_title(r"$\psi_{\tau}(t)$", fontsize=fontsize)
    ax[2, 1].set_xlabel("Task time", fontsize=fontsize)
    ax[2, 1].set_ylim(ylim3)

    baseline_losses = []
    control_losses = []
    for i, results in enumerate(result_manager3_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_losses.append(results.results["Loss_t_eq"])
        control_losses.append(results.results["Loss_t_control_opt"])

    mean_baseline_losses = np.mean(np.stack(baseline_losses, axis=0), axis=0)
    mean_control_losses = np.mean(np.stack(control_losses, axis=0), axis=0)
    std_baseline_losses = np.std(np.stack(baseline_losses, axis=0), axis=0)
    std_control_losses = np.std(np.stack(control_losses, axis=0), axis=0)

    ax[2, 0].plot(iters, mean_baseline_losses, 'k', lw=line_width, label="Baseline")
    ax[2, 0].plot(iters, mean_control_losses, 'C0', lw=line_width, label="Controlled")
    ax[2, 0].fill_between(iters, mean_baseline_losses - 2*std_baseline_losses, mean_baseline_losses + 2*std_baseline_losses,
                    color='k', alpha=0.3)
    ax[2, 0].fill_between(iters, mean_control_losses - 2*std_control_losses, mean_control_losses + 2*std_control_losses,
                    color='C0', alpha=0.3)
    ax[2, 0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[2, 0].set_ylabel("Vector", fontsize=fontsize)
    # ax[2, 0].set_title(r"$\mathcal{L}(t)$")
    ax[2, 0].spines[['right', 'top']].set_visible(False)
    ax[2, 0].text(-0.15, 1.05, subplot_labels[2, 0], transform=ax[2, 0].transAxes,
                  size=fontsize, weight='bold')
    #ax[2, 0].legend(fontsize=fontsize - 2, frameon=False)
    ax[2, 0].set_xlabel("Task time", fontsize=fontsize)


def single_neuron_param_plot(result_dict, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    figsize = plot_kwargs["figsize"]
    line_width = plot_kwargs["line_width"]
    selected_vars = plot_kwargs["selected_vars"]
    x_lim_vars = plot_kwargs["x_lim_vars"]
    var_label = plot_kwargs["var_label"]
    subplot_labels = plot_kwargs["subplot_labels"]
    n_plots = 0
    f, ax = plt.subplots(2, len(selected_vars), figsize=figsize)
    for i, var_name in enumerate(selected_vars):
        # CONTROL PLOT
        iters = [np.arange(len(data["data"].params["control_params"]["control"].time_span)) for data in result_dict[var_name]]
        control = [data["data"].params["control_params"]["control"] for data in result_dict[var_name]]
        n_curves = len(control)
        colors = cm.viridis(np.linspace(0, 1, n_curves))
        var_sweep = [data["run_vals"][var_name] for data in result_dict[var_name]]
        control = np.stack(control, axis=0)
        iters = np.array(iters, dtype=object)
        var_sweep = np.stack(var_sweep, axis=0)
        sort_index = np.argsort(var_sweep)
        control = control[sort_index, ...]
        var_sweep = var_sweep[sort_index, ...]
        iters = iters[sort_index, ...]

        for j in range(n_curves):
            label = var_label[i] + str(var_sweep[j])
            if var_name == "n_steps":
                iters[j] = iters[j]/np.amax(iters[j])
            if j == 0 or j==n_curves-1:
                ax[0, i].plot(iters[j], control[j].g.detach().cpu().numpy()[:, 0, 0], c=colors[j], label=label, lw=line_width)
            else:
                ax[0, i].plot(iters[j], control[j].g.detach().cpu().numpy()[:, 0, 0], c=colors[j], lw=line_width)
        ax[0, i].set_xlim(x_lim_vars[i])
        ax[0, i].spines[['right', 'top']].set_visible(False)
        ax[0, i].legend(fontsize=fontsize, frameon=False)
        ax[0, 0].set_ylabel(r"control signal $g(t)$", fontsize=fontsize)
        ax[0, i].tick_params(axis='both', which='major', labelsize=fontsize - 2)
        ax[0, i].text(-0.15, 1.05, subplot_labels[0, i], transform=ax[0, i].transAxes,
                size=fontsize, weight='bold')
        n_plots += 1
        if var_name == "betas":
            ax[0, 0].set_ylim(-0.02, 0.7)


        # LOSS PLOT
        iters = [np.arange(len(data["data"].params["control_params"]["control"].time_span)) for data in result_dict[var_name]]
        loss_eq = [data["data"].results["Loss_t_eq"] for data in result_dict[var_name]]
        loss_control = [data["data"].results["Loss_t_control_opt"] for data in result_dict[var_name]]
        control = [data["data"].params["control_params"]["control"] for data in result_dict[var_name]]

        n_curves = len(loss_eq)
        colors = cm.viridis(np.linspace(0, 1, n_curves))
        #if not isinstance(loss_eq, np.ndarray):
        #    loss_eq = [val.detach().cpu().numpy() for val in loss_eq]
        #if not isinstance(loss_control, np.ndarray):
        #    loss_control = [val.detach().cpu().numpy() for val in loss_control]

        var_sweep = [data["run_vals"][var_name] for data in result_dict[var_name]]
        cost_coef = np.array([data["data"].params["control_params"]["cost_coef"] for data in result_dict[var_name]])
        reward_convertion = np.array([data["data"].params["control_params"]["reward_convertion"] for data in result_dict[var_name]])

        loss_eq = np.array(loss_eq, dtype=object)
        loss_control = np.array(loss_control, dtype=object)
        control = np.stack(control, axis=0)
        iters = np.array(iters, dtype=object)
        var_sweep = np.stack(var_sweep, axis=0)
        sort_index = np.argsort(var_sweep)
        loss_eq = loss_eq[sort_index, ...]
        loss_control = loss_control[sort_index, ...]
        var_sweep = var_sweep[sort_index, ...]
        control = control[sort_index, ...]
        cost_coef = cost_coef[sort_index, ...]
        reward_convertion = reward_convertion[sort_index, ...]
        iters = iters[sort_index, ...]

        for j in range(n_curves):
            baseline_net_r = -control[j].reward_convertion*loss_eq[j]
            control[j].square_control_loss = False
            control_net_r = -control[j].reward_convertion*loss_control[j] - control[j].control_cost(get_numpy=True)
            net_diff = control_net_r - baseline_net_r
            if var_name == "n_steps":
                iters[j] = iters[j]/np.amax(iters[j])
            ax[1, i].plot(iters[j], net_diff, c=colors[j], lw=line_width)
        ax[1, i].set_xlim(x_lim_vars[i])
        ax[1, i].spines[['right', 'top']].set_visible(False)
        #ax[1, i].legend(fontsize=fontsize - 3)
        ax[1, 0].set_ylabel(r"$v_{C}(t) - v_{B}(t)$", fontsize=fontsize)
        ax[1, i].tick_params(axis='both', which='major', labelsize=fontsize - 2)
        ax[1, i].set_xlabel(r"Task time", fontsize=fontsize)
        if var_name == "n_steps":
            ax[1, i].set_xlabel(r"Normalized task time", fontsize=fontsize)
        ax[1, i].text(-0.15, 1.05, subplot_labels[1, i], transform=ax[1, i].transAxes,
                size=fontsize, weight='bold')
        n_plots += 1


def single_neuron_baseline_plot(result_manager, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    figsize = plot_kwargs["figsize"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]

    f, ax = plt.subplots(2, 2, figsize=figsize)

    ### INSTANT REWARD RATE ###
    iters = result_manager.results["iters"]
    baseline_iRR = result_manager.results["baseline_iRR"]
    control_iRR = result_manager.results["control_iRR"]

    ax[0, 0].plot(iters, baseline_iRR, "k", lw=line_width, label=r"Baseline")
    ax[0, 0].plot(iters, control_iRR, 'C0', lw=line_width, label=r"Control")
    ax[0, 0].legend(fontsize=fontsize, frameon=False)
    ax[0, 0].set_ylabel(r"$v(t)$", fontsize=fontsize)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[0, 0].text(-0.15, 1.05, subplot_labels[0, 0], transform=ax[0, 0].transAxes,
                  size=fontsize, weight='bold')
    ax[0, 0].spines[['right', 'top']].set_visible(False)

    ### LOSS FUNCTION ###
    iters = result_manager.results["iters"]
    iters_sim = result_manager.results["weights_iters_sim"]
    baseline_loss = result_manager.results["Loss_t_eq"]
    baseline_loss_sim = result_manager.results["Loss_t_sim"]
    control_loss = result_manager.results["Loss_t_control_opt"]
    control_loss_sim = result_manager.results["Loss_t_sim_OPT"]

    ax[0, 1].plot(iters, baseline_loss_sim, "k", lw=line_width-2, alpha=0.5)
    ax[0, 1].plot(iters, control_loss_sim, 'C0', lw=line_width-2, alpha=0.5)
    ax[0, 1].plot(iters, baseline_loss, "k", lw=line_width)
    ax[0, 1].plot(iters, control_loss, 'C0', lw=line_width)

    ax[0, 1].set_ylabel(r"$\mathcal{L}(t)$", fontsize=fontsize)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[0, 1].text(-0.15, 1.05, subplot_labels[0, 1], transform=ax[0, 1].transAxes,
                  size=fontsize, weight='bold')
    ax[0, 1].spines[['right', 'top']].set_visible(False)

    ### OPTIMAL G ###
    iters = result_manager.results["iters"]
    control_signal = result_manager.params["control_params"]["control"].g.detach().cpu().numpy()[:, 0, 0]

    ax[1, 0].plot(iters, control_signal, 'C0', lw=line_width)

    ax[1, 0].set_ylabel(r"Optimal $g(t)$", fontsize=fontsize)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[1, 0].text(-0.15, 1.05, subplot_labels[1, 0], transform=ax[1, 0].transAxes,
                  size=fontsize, weight='bold')
    ax[1, 0].set_xlabel("Task time", fontsize=fontsize)
    ax[1, 0].spines[['right', 'top']].set_visible(False)

    ### WEIGHTS ###
    iters = result_manager.results["iters"]
    baseline_weight = result_manager.results["W_t_eq"][:, 0, 0]
    control_weight = result_manager.results["W_t_control_opt"][:, 0, 0]
    controlled_mapping = control_weight*(1+control_signal)

    ax[1, 1].plot(iters, baseline_weight, "k", lw=line_width)
    ax[1, 1].plot(iters, control_weight, "C0", lw=line_width)
    ax[1, 1].plot(iters, controlled_mapping, 'C3--', lw=line_width, label=r"$\tilde{w}(t)$")

    ax[1, 1].legend(fontsize=fontsize, frameon=False)
    ax[1, 1].set_ylabel(r"$w(t)$", fontsize=fontsize)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[1, 1].text(-0.15, 1.05, subplot_labels[1, 1], transform=ax[1, 1].transAxes,
                  size=fontsize, weight='bold')
    ax[1, 1].set_xlabel("Task time", fontsize=fontsize)
    ax[1, 1].spines[['right', 'top']].set_visible(False)


def class_prop_plot(result_manager1_list, result_manager2_list, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    figsize = plot_kwargs["figsize"]
    line_width = plot_kwargs["line_width"]
    subplot_labels = plot_kwargs["subplot_labels"]
    analysis_path = plot_kwargs["analysis_path"]
    zoom_xlim = plot_kwargs["zoom_xlim"]
    zoom_ylim = plot_kwargs["zoom_ylim"]
    ax = plot_kwargs["ax"]
    window_size = 300
    xlim1 = None
    if "xlim1" in plot_kwargs.keys():
        xlim1 = plot_kwargs["xlim1"]

    extra_data = pd.read_pickle(analysis_path)

    ### LOSS PLOT ###
    loss_diff = []
    for i, results in enumerate(result_manager2_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_loss = results.results["Loss_t_eq"]
        control_loss = results.results["Loss_t_control_opt"]
        loss_diff.append(baseline_loss - control_loss)

    mean_loss_diff = np.mean(np.stack(loss_diff, axis=0), axis=0)
    std_loss_diff = np.std(np.stack(loss_diff, axis=0), axis=0, ddof=1)

    ax[0, 0].plot(iters, mean_loss_diff, 'C0', lw=line_width, label="Controlled")
    ax[0, 0].fill_between(iters, mean_loss_diff - 1*std_loss_diff,
                          mean_loss_diff + 1*std_loss_diff,
                          color='C0', alpha=0.3)

    #ax[0, 0].legend(fontsize=fontsize-2)
    # ax[0, 0].set_xlabel("Task time", fontsize=fontsize)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[0, 0].set_ylabel(r"$\mathcal{L}_{B}(t)-\mathcal{L}_{C}(t)$", fontsize=fontsize)
    ax[0, 0].set_title("MNIST", fontsize=fontsize)
    ax[0, 0].spines[['right', 'top']].set_visible(False)
    ax[0, 0].text(-0.15, 1.05, subplot_labels[0, 0], transform=ax[0, 0].transAxes,
                  size=fontsize, weight='bold')
    ax[0, 0].set_xlim(xlim1)

    loss_diff = []
    for i, results in enumerate(result_manager1_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_loss = results.results["Loss_t_eq"]
        control_loss = results.results["Loss_t_control_opt"]
        loss_diff.append(baseline_loss - control_loss)

    mean_loss_diff = np.mean(np.stack(loss_diff, axis=0), axis=0)
    std_loss_diff = np.std(np.stack(loss_diff, axis=0), axis=0)

    ax[0, 1].plot(iters, mean_loss_diff, 'C0', lw=line_width, label="Controlled")
    ax[0, 1].fill_between(iters, mean_loss_diff - 1 * std_loss_diff,
                          mean_loss_diff + 1 * std_loss_diff,
                          color='C0', alpha=0.3)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    # ax[0, 1].set_ylabel(r"$\mathcal{L}_{B}(t)-\mathcal{L}_{C}(t)$", fontsize=fontsize)
    ax[0, 1].set_title("Semantic", fontsize=fontsize)
    ax[0, 1].set_xlim([0, 13000])
    ax[0, 1].spines[['right', 'top']].set_visible(False)
    ax[0, 1].text(-0.15, 1.05, subplot_labels[0, 1], transform=ax[0, 1].transAxes,
                  size=fontsize, weight='bold')
    #
    ### NUS PLOT WITH TREE ###
    result_manager1 = result_manager1_list[0]
    result_manager2 = result_manager2_list[0]
    nus = result_manager1.results["final_engagement_coef"]
    iters = result_manager1.results["iters"]
    nus = nus.detach().cpu().numpy()
    n_levels = result_manager1.params["dataset_params"]["dataset"].h_level
    leaves_per_level = 2**(np.arange(n_levels))
    level_colors = ["C" + str(i) for i in range(n_levels)]
    level_per_curve = []
    current_level = 0
    for n_leaves in leaves_per_level:
        for i in range(n_leaves):
            level_per_curve.append(current_level)
        current_level += 1
    for i in range(nus.shape[-1]):
        ax[1, 1].plot(iters, nus[:, i], lw=line_width, color=level_colors[level_per_curve[i]])
    # ax[1, 1].set_xlabel("Task time", fontsize=fontsize)
    #ax[1, 1].set_ylabel(r"$\phi_{c}(t)$", fontsize=fontsize)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[1, 1].spines[['right', 'top']].set_visible(False)
    ax[1, 1].text(-0.15, 1.05, subplot_labels[1, 1], transform=ax[1, 1].transAxes,
                  size=fontsize, weight='bold')
    ax[1, 1].set_xlim([0, 13000])
    # ax[1, 1].set_xlim(xlim1)

    # Drawing hierarchy tree
    root_point = (9000, 1.9)
    height = 0.6
    width = 7000
    marker_size = 80
    draw_tree(ax[1, 1], n_levels, level_colors, root_point, height=height,
              width=width, marker_size=marker_size, line_width=line_width)
    #
    ### MNIST NUS PLOT ###
    iters = result_manager2.results["iters"]
    nus_list = []

    for i, results in enumerate(result_manager2_list):
        if i == 0:
            iters = results.results["iters"]
        nus = results.results["final_engagement_coef"]
        nus_list.append(nus.detach().cpu().numpy())

    mean_nus = np.mean(np.stack(nus_list, axis=0), axis=0)
    std_nus = np.std(np.stack(nus_list, axis=0), axis=0)

    for i in range(nus.shape[-1])[:10]:
        ax[1, 0].plot(iters, mean_nus[:, i], lw=line_width, label="Digit "+str(i))
    # ax[1, 1].legend()
    # ax[1, 0].set_xlabel("Task time", fontsize=fontsize)
    ax[1, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[1, 0].set_xlim(xlim1)
    #ax[1, 0].set_ylim([0, 0.0018])
    ax[1, 0].spines[['right', 'top']].set_visible(False)
    ax[1, 0].text(-0.15, 1.05, subplot_labels[1, 0], transform=ax[1, 0].transAxes,
                  size=fontsize, weight='bold')
    #ax[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #ax[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1, 0].set_ylabel(r"$\phi_{c}(t)$", fontsize=fontsize)

    current_digit = 0
    for p2 in range(2):
        for p1 in range(5):
            ax[1, 0].text(0.65*(1+p1*0.1), 0.8*(1-p2*0.18), str(current_digit), transform=ax[1, 0].transAxes,
                          size=fontsize+5, weight='bold', color="C"+str(current_digit))
            current_digit += 1

    # LOSS PER CLASS
    base_loss_per_class = extra_data["baseline_error_class"]
    control_loss_per_class = extra_data["control_error_class"]
    iters = extra_data["iters"]
    for i in range(base_loss_per_class.shape[1]):
        ax[2, 0].plot(iters, sliding_window(base_loss_per_class[:, i], half_window=window_size),
                      linestyle="--", lw=line_width-1.5,
                      color="C"+str(i), alpha=0.6)
        ax[2, 0].plot(iters, sliding_window(control_loss_per_class[:, i], half_window=window_size),
                      linestyle="-", lw=line_width-1.5,
                      color="C"+str(i), alpha=0.6)
    # ax[1, 1].set_xlabel("Task time", fontsize=fontsize)
    #ax[1, 1].set_ylabel(r"$\phi_{c}(t)$", fontsize=fontsize)
    ax[2, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[2, 0].spines[['right', 'top']].set_visible(False)
    ax[2, 0].text(-0.15, 1.05, subplot_labels[2, 0], transform=ax[2, 0].transAxes,
                  size=fontsize, weight='bold')
    ax[2, 0].set_ylabel(r"$\mathcal{L}_{c}(t)$", fontsize=fontsize)
    ax[2, 0].set_xlabel("Task time", fontsize=fontsize)
    ax[2, 0].set_xlim(xlim1)

    # COMPARED LOSS BETWEEN CURRICULUMS
    base_loss = extra_data["baseline_loss"]
    balanced_control_loss = extra_data["balanced_loss"]
    curriculum_loss = extra_data["curriculum_loss"]
    ax[2, 1].plot(iters, base_loss, 'C0', lw=line_width-1, label="Uniform")
    ax[2, 1].plot(iters, balanced_control_loss, 'C1', lw=line_width-1, label="Balanced")
    ax[2, 1].plot(iters, curriculum_loss, "C2", lw=line_width-1, label="Curriculum")
    # ax[1, 1].set_xlabel("Task time", fontsize=fontsize)
    # ax[1, 1].set_ylabel(r"$\phi_{c}(t)$", fontsize=fontsize)
    ax[2, 1]
    ax[2, 1].tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax[2, 1].spines[['right', 'top']].set_visible(False)
    ax[2, 1].text(-0.15, 1.05, subplot_labels[2, 1], transform=ax[2, 1].transAxes,
                  size=fontsize, weight='bold')
    ax[2, 1].set_ylabel(r"$\mathcal{L}(t)$", fontsize=fontsize)
    ax[2, 1].set_xlabel("Task time", fontsize=fontsize)
    ax[2, 1].set_xlim(zoom_xlim)
    ax[2, 1].set_ylim(zoom_ylim)
    ax[2, 1].legend(fontsize=fontsize-4, frameon=False)


def task_modulation_plot(results, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    figsize = plot_kwargs["figsize"]
    line_width = plot_kwargs["line_width"]
    ylim2 = plot_kwargs["ylim2"]
    ylim3 = plot_kwargs["ylim3"]
    subplot_labels = plot_kwargs["subplot_labels"]
    x_lim = None
    ax = plot_kwargs["ax"]
    if "x_lim" in plot_kwargs.keys():
        x_lim = plot_kwargs["x_lim"]

    ### LOSS PLOT
    baseline_losses = []
    iters = results.results["iters"]
    base_loss = results.results["Loss_t_eq"]
    control_loss = results.results["Loss_t_control_opt"]

    ax[0].plot(iters, base_loss, 'k', lw=line_width, label="Baseline")
    ax[0].plot(iters, control_loss, 'C0', lw=line_width, label="Controlled")
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[0].set_ylabel("Attentive", fontsize=fontsize)
    ax[0].set_title(r"$\mathcal{L}(t)$", fontsize=fontsize)
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].text(-0.15, 1.05, subplot_labels[0], transform=ax[0].transAxes,
                  size=fontsize, weight='bold')
    ax[0].legend(fontsize=fontsize - 3, frameon=False)

    ### Engagement plot ###
    phis = softmax(results.results["final_engagement_coef"].detach().cpu().numpy(), axis=1)
    colors = ["C"+str(i) for i in range(phis.shape[-1])]
    legend = ((0, 1), (7, 1), (8, 9))
    for i in range(phis.shape[-1]):
        leg = "Digits: " + str(legend[i])
        ax[1].plot(iters, phis[:, i], colors[i], lw=line_width, label=leg)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[1].legend(fontsize=fontsize - 2, frameon=False)
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].text(-0.15, 1.05, subplot_labels[1], transform=ax[1].transAxes,
                  size=fontsize, weight='bold')


def learning_rate_plot(manager_list, ax=None, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    skip_xlabel = plot_kwargs["skip_xlabel"]
    label_in_title = plot_kwargs["label_in_title"]
    subplot_labels = plot_kwargs["subplot_labels"]
    x_lim = None
    if "x_lim" in plot_kwargs.keys():
        x_lim = plot_kwargs["x_lim"]

    #############
    # Loss plot #
    #############
    plot_index = 1
    baseline_losses = []
    control_losses = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        baseline_losses.append(results.results["Loss_t_eq"])
        control_losses.append(results.results["Loss_t_control_opt"])

    mean_baseline_losses = np.mean(np.stack(baseline_losses, axis=0), axis=0)
    mean_control_losses = np.mean(np.stack(control_losses, axis=0), axis=0)
    std_baseline_losses = np.std(np.stack(baseline_losses, axis=0), axis=0)
    std_control_losses = np.std(np.stack(control_losses, axis=0), axis=0)

    ax[plot_index].plot(iters, mean_baseline_losses, 'k', lw=line_width, label="Baseline")
    ax[plot_index].plot(iters, mean_control_losses, 'C0', lw=line_width, label="Controlled")
    ax[plot_index].fill_between(iters, mean_baseline_losses - 2*std_baseline_losses, mean_baseline_losses + 2*std_baseline_losses,
                    color='k', alpha=0.3)
    ax[plot_index].fill_between(iters, mean_control_losses - 2*std_control_losses, mean_control_losses + 2*std_control_losses,
                    color='C0', alpha=0.3)
    if not skip_xlabel:
        ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    ax[plot_index].set_xlim(x_lim)
    if label_in_title:
        ax[plot_index].set_title(r"$\mathcal{L}(t)$", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel(r"$\mathcal{L}(t)$", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')

    # ax[plot_index].legend(fontsize=fontsize - 2)

    ###########################
    # Net instant reward rate #
    ###########################
    plot_index = 0
    baseline_r = []
    control_r = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        opt_lr = results.results["control_signal"].detach().cpu().numpy()
        cost_coef = results.params["control_params"]["cost_coef"]
        reward_conv = results.params["control_params"]["reward_convertion"]
        control_cost = cost_coef * opt_lr ** 2
        r_control = -reward_conv*results.results["Loss_t_control_opt"] - control_cost
        r_base = -reward_conv*results.results["Loss_t_eq"]
        baseline_r.append(r_base)
        control_r.append(r_control)

    mean_baseline_r = np.mean(np.stack(baseline_r, axis=0), axis=0)
    mean_control_r = np.mean(np.stack(control_r, axis=0), axis=0)
    std_baseline_r = np.std(np.stack(baseline_r, axis=0), axis=0)
    std_control_r = np.std(np.stack(control_r, axis=0), axis=0)

    ax[plot_index].plot(iters, mean_baseline_r, 'k', lw=line_width, label="Baseline")
    ax[plot_index].plot(iters, mean_control_r, 'C0', lw=line_width, label="Control")
    ax[plot_index].fill_between(iters, mean_baseline_r - 2*std_baseline_r, mean_baseline_r + 2*std_baseline_r,
                    color='k', alpha=0.3)
    ax[plot_index].fill_between(iters, mean_control_r - 2*std_control_r, mean_control_r + 2*std_control_r,
                    color='C0', alpha=0.3)
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-2, frameon=False)
    if not skip_xlabel:
        ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title(r"$v(t)$", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel(r"$v(t)$", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')
    # ax[plot_index].text(.01, .99, '(a)', ha='left', va='top', transform=ax[plot_index].transAxes)

    ###################
    # Weight sparsity #
    ###################
    plot_index = 2
    baseline_l1 = []
    baseline_l2 = []
    control_l1 = []
    control_l2 = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        W1_t, W2_t = results.results["W1_t_eq"], results.results["W2_t_eq"]
        W1_control, W2_control = results.results["W1_t_control_opt"], results.results["W2_t_control_opt"]
        base_l1, base_l2 = compute_all_weight_sparsity([W1_t, W2_t])
        con_l1, con_l2 = compute_all_weight_sparsity([W1_control, W2_control])
        baseline_l1.append(base_l1)
        baseline_l2.append(base_l2)
        control_l1.append(con_l1)
        control_l2.append(con_l2)

    baseline_l1 = np.mean(np.stack(baseline_l1, axis=0), axis=0)
    baseline_l2 = np.mean(np.stack(baseline_l2, axis=0), axis=0)
    control_l1 = np.mean(np.stack(control_l1, axis=0), axis=0)
    control_l2 = np.mean(np.stack(control_l2, axis=0), axis=0)
    ax[plot_index].plot(iters, baseline_l1, 'k', lw=line_width, label=r"Base $L1$")
    ax[plot_index].plot(iters, control_l1, 'C0', lw=line_width, label=r"Ctrl $L1$")
    ax[plot_index].plot(iters, baseline_l2, 'k--', lw=line_width, label=r"Base $L2$")
    ax[plot_index].plot(iters, control_l2, 'C0--', lw=line_width, label=r"Ctrl $L2$")
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-5, frameon=False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title("Weight norms", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("Weight norms", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')

    ################
    # Control Size #
    ################
    plot_index = 3
    normalize = True
    base_loss_der = []
    ctrl_loss_der = []
    G1_size = []
    G2_size = []
    for i, results in enumerate(manager_list):
        if i == 0:
            iters = results.results["iters"]
        G1_tilda, G2_tilda = results.results["control_signal"]
        G1_tilda, G2_tilda = G1_tilda.detach().cpu().numpy(), G2_tilda.detach().cpu().numpy()
        G1_t = G1_tilda - np.ones(G1_tilda.shape)
        G2_t = G2_tilda - np.ones(G2_tilda.shape)
        _, G1_norm = compute_weight_sparsity(G1_t)
        _, G2_norm = compute_weight_sparsity(G2_t)
        G1_size.append(G1_norm)
        G2_size.append(G2_norm)
        b_der_loss = np.diff(results.results["Loss_t_eq"])
        c_der_loss = np.diff(results.results["Loss_t_control_opt"])
        b_der_loss = np.concatenate([b_der_loss, (b_der_loss[-1],)])
        c_der_loss = np.concatenate([c_der_loss, (c_der_loss[-1],)])
        base_loss_der.append(b_der_loss)
        ctrl_loss_der.append(c_der_loss)

    base_loss_der = np.mean(np.stack(base_loss_der, axis=0), axis=0)
    ctrl_loss_der = np.mean(np.stack(ctrl_loss_der, axis=0), axis=0)
    G1_size = np.mean(np.stack(G1_size, axis=0), axis=0)
    G2_size = np.mean(np.stack(G2_size, axis=0), axis=0)
    if normalize:
        base_loss_der = -base_loss_der / np.max(np.abs(base_loss_der))
        ctrl_loss_der = -ctrl_loss_der / np.max(np.abs(ctrl_loss_der))
        G1_size = G1_size / np.max(np.abs(G1_size))
        G2_size = G2_size / np.max(np.abs(G2_size))
    ax[plot_index].plot(iters, base_loss_der, 'k', lw=line_width, label=r"Base $d\mathcal{L}/dt$")
    ax[plot_index].plot(iters, ctrl_loss_der, 'C0', lw=line_width, label=r"Ctrl $d\mathcal{L}/dt$")
    ax[plot_index].plot(iters, G1_size, 'C2--', lw=line_width-1, label=r"$G_1(t)$ size")
    ax[plot_index].plot(iters, G2_size, 'C3--', lw=line_width-1, label=r"$G_2(t)$ size")
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-5, frameon=False)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title("Normalized unit", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("Normalized unit", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax[plot_index].spines[['right', 'top']].set_visible(False)
    ax[plot_index].text(-0.15, 1.05, subplot_labels[plot_index], transform=ax[plot_index].transAxes,
                  size=fontsize, weight='bold')

    return ax

def compute_control_cost(G1_t, G2_t=None, cost_coef=0.3):
    if G2_t is None:
        control_cost = np.exp(cost_coef * (np.sum(G1_t ** 2, axis=(-1, -2)))) - 1
    else:
        control_cost = np.exp(cost_coef * (np.sum(G1_t ** 2, axis=(-1, -2)) + np.sum(G2_t ** 2, axis=(-1, -2)))) - 1
    return control_cost


def compute_weight_sparsity(W_t):
    W1_norm1 = np.mean(np.abs(W_t), axis=(-1, -2))
    W1_norm2 = np.sqrt(np.mean(W_t**2, axis=(-1, -2)))
    return W1_norm1, W1_norm2


def compute_all_weight_sparsity(w_list):
    flatten_w = [np.reshape(w, newshape=(w.shape[0], -1)) for w in w_list]
    conc_w = np.concatenate(flatten_w, axis=1)
    W_norm1 = np.mean(np.abs(conc_w), axis=-1)
    W_norm2 = np.sqrt(np.mean(conc_w**2, axis=-1))
    return W_norm1, W_norm2


def sliding_window(vals, half_window=10):
    smooth_vals = []
    for i in range(len(vals)):
        min_index = np.max([i-half_window, 0])
        max_index = np.min([i+half_window, len(vals)])
        smooth_vals.append(np.mean(vals[min_index:max_index]))
    return np.array(smooth_vals)


def draw_tree(ax, n_levels, level_colors, root_point, height=3.0, width=3.0, marker_size=500, line_width=3):
    height_per_level = height / (n_levels - 1)
    width_per_level = width / (2 ** (np.arange(n_levels) + 1))
    leafs_per_level = 2 ** np.arange(n_levels)
    level_points = {}

    for level in range(n_levels):
        if level == 0:
            ax.scatter(root_point[0], root_point[1], c=level_colors[level], s=marker_size, zorder=2)
            level_points[level] = [root_point, ]
        else:
            level_width = width_per_level[level]
            new_y = level_points[level - 1][0][1] - height_per_level
            this_level_points = []
            for pre_point in level_points[level - 1]:
                left_point = (pre_point[0] - level_width, new_y)
                right_point = (pre_point[0] + level_width, new_y)
                this_level_points.append(left_point)
                this_level_points.append(right_point)
                ax.plot([left_point[0], pre_point[0]], [left_point[1], pre_point[1]], color="k", lw=line_width,
                        zorder=0)
                ax.plot([right_point[0], pre_point[0]], [right_point[1], pre_point[1]], color="k", lw=line_width,
                        zorder=0)
                ax.scatter(left_point[0], left_point[1], s=marker_size, c=level_colors[level], zorder=2)
                ax.scatter(right_point[0], right_point[1], s=marker_size, c=level_colors[level], zorder=2)
            level_points[level] = this_level_points

