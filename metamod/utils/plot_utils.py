import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.palettes import Viridis, Category10, Category20
from bokeh.io import export_svg

import metamod.utils


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
    x_lim = None
    if "x_lim" in plot_kwargs.keys():
        x_lim = plot_kwargs["x_lim"]

    if ax is None:
        f, ax = plt.subplots(1, 3, figsize=(12, 5))

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
        ax[plot_index].set_title("$\mathcal{L}(t)$", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("$\mathcal{L}(t)$", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
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
    ax[plot_index].legend(fontsize=fontsize-2)
    if not skip_xlabel:
        ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title("$v(t) = -\eta \mathcal{L}(t) - C(G(t))$", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("$v(t) = -\eta \mathcal{L}(t) - C(G(t))$", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)
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
    ax[plot_index].plot(iters, baseline_l1, 'k', lw=line_width, label="Base $L1$")
    ax[plot_index].plot(iters, control_l1, 'C0', lw=line_width, label="Ctrl $L1$")
    ax[plot_index].plot(iters, baseline_l2, 'k--', lw=line_width, label="Base $L2$")
    ax[plot_index].plot(iters, control_l2, 'C0--', lw=line_width, label="Ctrl $L2$")
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-2)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title("Weight norms", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("Weight norms", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)

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
    ax[plot_index].plot(iters, base_loss_der, 'k', lw=line_width, label="Base $d\mathcal{L}/dt$")
    ax[plot_index].plot(iters, ctrl_loss_der, 'C0', lw=line_width, label="Ctrl $d\mathcal{L}/dt$")
    ax[plot_index].plot(iters, G1_size, 'C2--', lw=line_width-1, label="$G_1(t)$ size")
    ax[plot_index].plot(iters, G2_size, 'C3--', lw=line_width-1, label="$G_2(t)$ size")
    ax[plot_index].set_xlim(x_lim)
    ax[plot_index].legend(fontsize=fontsize-2)
    ax[plot_index].set_xlabel("Task time", fontsize=fontsize)
    if label_in_title:
        ax[plot_index].set_title("Normalized unit", fontsize=fontsize)
    else:
        ax[plot_index].set_ylabel("Normalized unit", fontsize=fontsize)
    ax[plot_index].tick_params(axis='both', which='major', labelsize=fontsize-2)

    return ax


def task_switch_plot(result_manager, **plot_kwargs):
    fontsize = plot_kwargs["fontsize"]
    line_width = plot_kwargs["line_width"]
    figsize = plot_kwargs["figsize"]
    zoom_xlim = plot_kwargs["zoom_xlim"]
    zoom_ylim = plot_kwargs["zoom_ylim"]

    fig = plt.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.25)  # 2x2 grid
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
    ax0.legend(fontsize=fontsize-2)
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
    ax0.plot([zoom_xlim[0], 12800], [zoom_ylim[0], -0.8], "C3--", lw=line_width, alpha=alpha, zorder=0, clip_on=False)
    ax0.plot([zoom_xlim[1], np.amax(iters)], [zoom_ylim[0], -0.8], "C3--", lw=line_width, alpha=alpha, zorder=0, clip_on=False)
    ax0.set_ylabel("$\mathcal{L}(t)$", fontsize=fontsize)


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
    ax2.legend(fontsize=fontsize-2)
    ax2.set_xlabel("Task time", fontsize=fontsize)
    ax2.set_xlim(zoom_xlim)
    ax2.set_ylim(zoom_ylim)
    ax2.legend(fontsize=fontsize-2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize - 2)

    ################################
    # Loss peaks and control peaks #
    ################################
    ax1.plot(iters[peak_iters], loss_t_eq[peak_iters], 'k-o', lw=line_width)#, label="Baseline $\mathcal{L}(t)$")
    ax1.plot(iters[peak_iters], loss_t_control[peak_iters], 'C0-o', lw=line_width)#, label="Control $\mathcal{L}(t)$")
    ax1.plot(iters[peak_iters], G_cost[peak_iters], 'C2-o', lw=line_width) #label="Normalized $C(G(t))$")
    # ax1.legend(fontsize=fontsize-2)
    ax1.set_xlabel("Task time", fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax1.set_ylabel("Values at switch time", fontsize=fontsize)


def compute_control_cost(G1_t, G2_t, cost_coef):
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



