import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter1d

# matplotlib.use('Agg')  # No GUI
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from src.conf import EVAL_ROUND
from src.utils import verify_metrics, load


def w_heatmaps(W1, W2, threshold=0.001):
    """
    Draws light-colored heatmaps of two doubly stochastic matrices W1 and W2.
    Elements below the threshold are displayed as black cells with white text.
    The cells have a rectangle shape (length = 2 * width).

    Parameters:
        W1 (numpy.ndarray): The first doubly stochastic matrix.
        W2 (numpy.ndarray): The second doubly stochastic matrix.
        threshold (float): Threshold below which elements are shown in black.
    """
    if not isinstance(W1, np.ndarray) or not isinstance(W2, np.ndarray):
        raise ValueError("Inputs must be NumPy arrays.")
    if W1.shape != W2.shape or len(W1.shape) != 2 or W1.shape[0] != W1.shape[1]:
        raise ValueError("Inputs must be square matrices of the same size.")

    # Create a custom light colormap for normal values
    light_cmap = ListedColormap(["#f8f8ff", "#d3d3d3", "#b0c4de", "#add8e6", "#87ceeb", "#4682b4"])

    # Prepare masks for elements below the threshold
    mask1 = W1 < threshold
    masked_W1 = np.ma.masked_where(mask1, W1)

    mask2 = W2 < threshold
    masked_W2 = np.ma.masked_where(mask2, W2)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))  # Side-by-side heatmaps
    ax1, ax2 = axes

    # Plot for W1
    ax1.imshow(masked_W1, cmap=light_cmap, interpolation='nearest')
    ax1.imshow(mask1, cmap=ListedColormap(['black']), interpolation='nearest', alpha=mask1 * 1.0)
    ax1.set_aspect(0.5)  # Rectangular cells (length = 2 * width)
    ax1.axis('off')
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            value = W1[i, j]
            if value < threshold:
                ax1.text(j, i, f"{value:.3f}", ha='center', va='center', color='white', fontsize=8)
            else:
                ax1.text(j, i, f"{value:.3f}", ha='center', va='center', color='black', fontsize=8)
    ax1.text(4, 11, '$W$', fontsize=20, weight='bold')

    # Plot for W2
    ax2.imshow(masked_W2, cmap=light_cmap, interpolation='nearest')
    ax2.imshow(mask2, cmap=ListedColormap(['black']), interpolation='nearest', alpha=mask2 * 1.0)
    ax2.set_aspect(0.5)  # Rectangular cells (length = 2 * width)
    ax2.axis('off')
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            value = W2[i, j]
            if value < threshold:
                ax2.text(j, i, f"{value:.3f}", ha='center', va='center', color='white', fontsize=8)
            else:
                ax2.text(j, i, f"{value:.3f}", ha='center', va='center', color='black', fontsize=8)
    ax2.text(4, 11, '$W_p$', fontsize=20, weight='bold', horizontalalignment='left', verticalalignment='bottom')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_train_time(logs_time, metric='accuracy', measure="mean", info=None, plot_peer=None):
    logs, times = logs_time
    if isinstance(logs, str):
        logs = load(logs)
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    # prepare data
    std_data = None
    if measure == "mean":
        data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        print(data)
    elif measure == "mean-std":
        if plot_peer is None:
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            print(f">>>>> Plotting chart for Peer({plot_peer})...")
            data = [v[metric] for v in logs[plot_peer]]
        std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    elif measure == "max":
        data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    else:
        data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    # plot data
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric.capitalize()}'
    # title = f'{_metric.capitalize()} vs. No. of rounds'
    title = None
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    # x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    x = times
    # Configs
    plt.grid(linestyle='dashed')
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    # Plot
    plt.plot(x, data)
    plt.scatter(x, data, color='red')
    if std_data is not None:
        plt.fill_between(x, data - std_data, data + std_data, alpha=.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=9)

    plt.show()


def plot_train_history(logs, metric='accuracy', measure="mean", info=None, plot_peer=None):
    if isinstance(logs, str):
        logs = load(logs)
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    # prepare data
    std_data = None
    if measure == "mean":
        data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    elif measure == "mean-std":
        if plot_peer is None:
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            print(f">>>>> Plotting chart for Peer({plot_peer})...")
            data = [v[metric] for v in logs[plot_peer]]
        std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    elif measure == "max":
        data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    else:
        data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
    # plot data
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric.capitalize()}'
    # title = f'{_metric.capitalize()} vs. No. of rounds'
    title = None
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
    # Configs
    plt.grid(linestyle='dashed')
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    # Plot
    plt.plot(x, data)
    if std_data is not None:
        plt.fill_between(x, data - std_data, data + std_data, alpha=.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title, fontsize=9)

    plt.show()


"""
    X ---> 100
    a ---> 100-86
"""


def plot_trans_energy():
    total_energy = {  # mnist
        "$W, \gamma=1$": 0.10686077166677443 * 10,
        "$W_p, \gamma=1$": 0.0676431849027338 * 10,
        "$W_p, \gamma=0.5$": 0.05357876031899707 * 10,
        "$W_p, \gamma=0.4$": 0.05893663635089678 * 10,
        "$W_p, \gamma=0.3$": 0.07634973345457081 * 10,
        "$W_p, \gamma=0.2$": 0.06697345039874635 * 10,
        "$W_p, \gamma=0.1$": 0.03348672519937317 * 10,
    }
    # total_energy = {  # cifar
    #     "$W, \gamma=1$": 9.86,
    #     "$W_p, \gamma=1$": 8.56,
    #     "$W_p, \gamma=0.5$": 6.88,
    #     "$W_p, \gamma=0.4$": 6.18,
    #     "$W_p, \gamma=0.3$": 9.27,
    #     "$W_p, \gamma=0.2$": 7.73,
    #     "$W_p, \gamma=0.1$": 4.3080,
    # }
    labels = list(total_energy.keys())
    values = list(total_energy.values())
    plt.figure()
    colors = ['lightgreen', 'skyblue', 'skyblue', 'skyblue', 'skyblue', 'skyblue', 'red']
    # vals = ['100\nRounds', '100\nRounds', '160\nRounds', '220\nRounds', '380\nRounds', '500\nRounds', '500 \nRounds']
    vals = ['90\nRounds', '100\nRounds', '160\nRounds', '180\nRounds', '360\nRounds', '450\nRounds', '500 \nRounds']
    bars = plt.bar(labels, values, color=colors, edgecolor='black')
    for bar, value in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height() / 2,  # Vertical position inside the bar
            f"{value}",  # Format the value to 4 decimal places
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            color='black',  # Text color
            fontsize=10
        )

    # Adding title and labels
    plt.grid(linestyle='dashed')
    plt.xlabel("Configurations", fontsize=14)
    plt.ylabel("Transmission Energy [J]", fontsize=13)
    # Customize x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=13)
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Show the plot
    plt.show()


def plot_compu_energy():
    # total_energy = { # mnist
    #     "$W, \gamma=1$": 0.10686077166677443,
    #     "$W_p, \gamma=1$": 0.0676431849027338,
    #     "$W_p, \gamma=0.5$": 0.05357876031899707,
    #     "$W_p, \gamma=0.4$": 0.05893663635089678,
    #     "$W_p, \gamma=0.3$": 0.07634973345457081,
    #     "$W_p, \gamma=0.2$": 0.06697345039874635,
    #     "$W_p, \gamma=0.1$": 0.03348672519937317,
    # }
    total_energy = {  # cifar
        "$W, \gamma=1$": 9.86,
        "$W_p, \gamma=1$": 8.56,
        "$W_p, \gamma=0.5$": 6.88,
        "$W_p, \gamma=0.4$": 6.18,
        "$W_p, \gamma=0.3$": 9.27,
        "$W_p, \gamma=0.2$": 7.73,
        "$W_p, \gamma=0.1$": 4.3080,
    }
    labels = list(total_energy.keys())
    values = list(total_energy.values())
    plt.figure()
    colors = ['lightgreen', 'skyblue', 'skyblue', 'skyblue', 'skyblue', 'skyblue', 'red']
    # vals = ['100\nRounds', '100\nRounds', '160\nRounds', '220\nRounds', '380\nRounds', '500\nRounds', '500 \nRounds']
    vals = ['90\nRounds', '100\nRounds', '160\nRounds', '180\nRounds', '360\nRounds', '450\nRounds', '500 \nRounds']
    bars = plt.bar(labels, values, color=colors, edgecolor='black')
    for bar, value in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height() / 2,  # Vertical position inside the bar
            f"{value}",  # Format the value to 4 decimal places
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            color='black',  # Text color
            fontsize=10
        )

    # Adding title and labels
    plt.grid(linestyle='dashed')
    plt.xlabel("Configurations", fontsize=14)
    plt.ylabel("Transmission Energy [J]", fontsize=13)
    # Customize x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=13)
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Show the plot
    plt.show()


def plot_many(metric='accuracy', measure="mean", info=None):
    # data = load("dfl_log_10_101_opt_True_238.pkl")["energy"]
    # print(sum([x[0] for x in data.values()]) / 101 * 101)
    # print(sum([x[1] * 100 / 18 for x in data.values()]) / 101 * 500 * 0.1)
    # exit(0)

    # mnist
    # gamma00 = load("dfl_log_10_101_opt_False_250.pkl")["train"]
    # gamma10 = load("dfl_log_10_101_opt_True_gamma_1_250.pkl")["train"]
    # gamma01 = load("dfl_log_10_1001_opt_True_gamma0.1_420.pkl")["train"]
    # gamma02 = load("dfl_log_10_1001_opt_True_gamma0.2_420.pkl")["train"]
    # gamma03 = load("dfl_log_10_1001_opt_True_gamma0.3_420.pkl")["train"]
    # gamma04 = load("dfl_log_10_1001_opt_True_gamma0.4_420.pkl")["train"]
    # gamma05 = load("dfl_log_10_1001_opt_True_gamma0.5_420.pkl")["train"]

    # cifar
    gamma00 = load("cifar/cifar_10_501_opt_False_False_gamma_1_449.pkl")["train"]
    gamma10 = load("cifar/cifar_10_501_opt_True_True_gamma_1.0_869.pkl")["train"]
    gamma01 = load("cifar/cifar_10_501_opt_True_True_gamma_0.1_157.pkl")["train"]
    gamma01 = load("cifar/cifar_10_501_opt_True_True_gamma_opt_157.pkl")["train"]
    gamma02 = load("cifar/cifar_10_501_opt_True_True_gamma_0.2_157.pkl")["train"]
    gamma03 = load("cifar/cifar_10_501_opt_True_True_gamma_0.3_157.pkl")["train"]
    gamma04 = load("cifar/cifar_10_501_opt_True_True_gamma_0.4_157.pkl")["train"]
    gamma05 = load("cifar/cifar_10_501_opt_True_True_gamma_0.5_157.pkl")["train"]

    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    data00 = np.mean([[v[metric] for v in lo] for lo in gamma00.values()], axis=0)
    data00 = gaussian_filter1d(data00, sigma=1)
    std_data00 = np.std([[v[metric] for v in lo] for lo in gamma00.values()], axis=0)
    lnd00 = len(data00)
    x00 = range(0, lnd00 * EVAL_ROUND, EVAL_ROUND)
    data10 = np.mean([[v[metric] for v in lo] for lo in gamma10.values()], axis=0)
    data10 = gaussian_filter1d(data10, sigma=1)
    std_data10 = np.std([[v[metric] for v in lo] for lo in gamma10.values()], axis=0)
    lnd10 = len(data10)
    x10 = range(0, lnd10 * EVAL_ROUND, EVAL_ROUND)
    data01 = np.mean([[v[metric] for v in lo] for lo in gamma01.values()], axis=0)
    data01 = gaussian_filter1d(data01, sigma=1)
    lnd01 = len(data01)  # MNIST 50 | CIFAR  # len(data01)
    x01 = range(0, lnd01 * EVAL_ROUND, EVAL_ROUND)
    data02 = np.mean([[v[metric] for v in lo] for lo in gamma02.values()], axis=0)
    data02 = gaussian_filter1d(data02, sigma=1)
    lnd02 = len(data02)  # MNIST 50 | CIFAR 501 # len(data02)
    x02 = range(0, lnd02 * EVAL_ROUND, EVAL_ROUND)
    data03 = np.mean([[v[metric] for v in lo] for lo in gamma03.values()], axis=0)
    data03 = gaussian_filter1d(data03, sigma=1)
    lnd03 = len(data03)  # MNIST 38 | CIFAR 360 # len(data03)
    x03 = range(0, lnd03 * EVAL_ROUND, EVAL_ROUND)
    data04 = np.mean([[v[metric] for v in lo] for lo in gamma04.values()], axis=0)
    data04 = gaussian_filter1d(data04, sigma=1)
    lnd04 = len(data04)  # MNIST 22 | CIFAR 190 # len(data04)
    x04 = range(0, lnd04 * EVAL_ROUND, EVAL_ROUND)
    data05 = np.mean([[v[metric] for v in lo] for lo in gamma05.values()], axis=0)
    data05 = gaussian_filter1d(data05, sigma=1)
    lnd05 = len(data05)  # MNIST 16 | CIFAR 170 # len(data05)
    x05 = range(0, lnd05 * EVAL_ROUND, EVAL_ROUND)
    # plot data
    xlabel = 'Number of rounds'
    ylabel = f'Test Accuracy'
    title = None
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    # , color=colors[i], label=mean[i][1], linestyle=line_styles[i]
    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    colors = ["#000", "#ff7f0e", "#2ca02c", "#17becf", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X"]
    plt.plot(x00, data00[:lnd00], label="Baseline", color=colors[0], marker=markers[0], markersize=4, linewidth=3)
    plt.plot(x10, data10[:lnd10], label="$W_p, \gamma=1$", color=colors[1], marker=markers[1], markersize=3)
    plt.plot(x01, data01[:lnd01], label="$W_p, \gamma=0.1$", color=colors[2], marker=markers[2], markersize=3)
    plt.plot(x02, data02[:lnd02], label="$W_p, \gamma=0.2$", color=colors[3], marker=markers[3], markersize=3)
    plt.plot(x03, data03[:lnd03], label="$W_p, \gamma=0.3$", color=colors[4], marker=markers[4], markersize=3)
    plt.plot(x04, data04[:lnd04], label="$W_p, \gamma=0.4$", color=colors[5], marker=markers[5], markersize=3)
    plt.plot(x05, data05[:lnd05], label="$W_p, \gamma=0.5$", color=colors[6], marker=markers[6], markersize=3)

    # plt.plot(x01, data00[:lnd01], label="Without network optimization", linestyle='-.')  # , '-x'
    # plt.fill_between(x01, data00[:lnd01] - std_data00[:lnd01], data00[:lnd01] + std_data00[:lnd01], alpha=.1)
    # plt.plot(x02, data10[:lnd02], label="With network optimization", linestyle='-')
    # plt.fill_between(x02, data10[:lnd02] - std_data10[:lnd02], data10[:lnd02] + std_data10[:lnd02], alpha=.1)
    # Configs
    plt.grid(linestyle='dashed')
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    if title is not None:
        plt.title(title, fontsize=9)
    plt.legend(markerscale=2)
    plt.show()


def plot_manymore(exps, metric='accuracy', measure="mean", info=None, save=False):
    # Configs
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    xlabel = 'Rounds'
    ylabel = f' {measure.capitalize()} {_metric.capitalize()}'
    title = f'{_metric.capitalize()} vs. No. of rounds'
    if info is not None:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel(xlabel, fontsize=13)
    colors = ['green', 'blue', 'orange', 'black', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']
    # colors = ['black', 'green', 'orange', 'blue', 'red', 'grey', 'tan', 'pink', 'navy', 'aqua']
    line_styles = ['-', '--', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':']
    plt.grid(linestyle='dashed')
    plt.rc('legend', fontsize=12)
    plt.xticks(fontsize=13, )
    plt.yticks(fontsize=13, )
    std_data = None
    for i, exp in enumerate(exps):
        # Data
        logs = load(exp['file'])
        name = exp.get('name', "")
        if measure == "mean":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "mean-std":
            data = np.mean([[v[metric] for v in lo] for lo in logs.values()], axis=0)
            std_data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        elif measure == "max":
            data = np.max([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        else:
            data = np.std([[v[metric] for v in lo] for lo in logs.values()], axis=0)
        x = range(0, len(data) * EVAL_ROUND, EVAL_ROUND)
        plt.plot(x, data, color=colors[i], label=name, linestyle=line_styles[i])
        if std_data is not None:
            plt.fill_between(x, data - std_data, data + std_data, color=colors[i], alpha=.1)

    plt.legend(loc="lower right", shadow=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title(title)
    if save:
        unique = np.random.randint(100, 999)
        plt.savefig(f"../out/EXP_{unique}.pdf")
    plt.show()


# ===***===***===***===***===***===***===***===***===***===***===***===***===***===***===***===***

def plot_w_wp(metric='accuracy', measure="mean", info=None):
    no_op = load("dfl_log_10_101_opt_False_gamma_1_250.pkl")["train"]
    ye_op = load("dfl_log_10_101_opt_True_gamma_1_250.pkl")["train"]
    # get correct metrics
    _metric = metric
    metric, measure = verify_metrics(metric, measure)
    no_data = np.mean([[v[metric] for v in lo] for lo in no_op.values()], axis=0)
    ye_data = np.mean([[v[metric] for v in lo] for lo in ye_op.values()], axis=0)
    noo_std = np.std([[v[metric] for v in lo] for lo in no_op.values()], axis=0)
    yes_std = np.std([[v[metric] for v in lo] for lo in ye_op.values()], axis=0)
    # plot data
    xlabel = 'Number of rounds'
    ylabel = f'Test Accuracy'
    title = None
    if info:
        xlabel = info.get('xlabel', xlabel)
        ylabel = info.get('ylabel', ylabel)
        title = info.get('title', title)
    x = range(0, len(no_data) * EVAL_ROUND, EVAL_ROUND)
    # , color=colors[i], label=mean[i][1], linestyle=line_styles[i]
    plt.plot(x, no_data, label="Without network optimization", linestyle='-.')  # , '-x'
    plt.fill_between(x, no_data - noo_std, no_data + noo_std, alpha=.1)
    plt.plot(x, ye_data, label="With network optimization", linestyle='-')
    plt.fill_between(x, ye_data - yes_std, ye_data + yes_std, alpha=.1)
    # Configs
    plt.grid(linestyle='dashed')
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    if title is not None:
        plt.title(title, fontsize=9)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example doubly stochastic matrices
    # plot_many()
    plot_trans_energy()
    # plot_compu_energy()
    # plot_total_energy()
