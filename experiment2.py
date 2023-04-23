import numpy as np
import matplotlib.pyplot as plt
import pyvis
import networkx as nx
from math import comb
from architecture.MLP import FCNet
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.scale import LogScale
import torch
from matplotlib.colors import LogNorm


"""
# visualizing the loss barrier after aligning the models

# create 2*5 subplots with enough space between them
fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=True)

# for widths [3, 4, 5, 6, 12], load the loss barrier and show heatmap, histogram
for i, width in enumerate([3, 4, 5, 6, 12]):
    # load loss barrier
    loss_barrier = np.load(f"logs\sigmoid\gaussian\loss_barriers_s512_w{width}_d1.npy")
    # show heatmap with colorbar
    ax = axes[0, i]
    im = ax.imshow(loss_barrier, cmap="jet", vmin=0, vmax=4)
    ax.set_title("width = {}".format(width))
    # show histogram
    ax = axes[1, i]
    ax.hist(loss_barrier.flatten(), bins=100, range=(0, 8))
    # draw vertical line at 5, 50, 95 percentile
    for p in [5, 50, 95]:
        ax.axvline(np.percentile(loss_barrier, p), color="k", linestyle="--")
    ax.set_title("width = {}".format(width))
    ax.set_xlabel("loss barrier")
# show colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

# save figure
plt.savefig("gaussian_loss_barrier_s512_d1.png", dpi=300)
plt.show()
"""

"""
# visualizing the loss barrier before aligning the models

# create 2*5 subplots with enough space between them
fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=True)

# for widths [3, 4, 5, 6, 12], load the loss barrier and show heatmap, histogram
for i, width in enumerate([3, 4, 5, 6, 12]):
    # load loss barrier
    loss_barrier = np.load(
        f"logs/sigmoid/gaussian/naive_loss_barriers_s512_w{width}_d1.npy"
    )
    # show heatmap with colorbar
    ax = axes[0, i]
    im = ax.imshow(loss_barrier, cmap="jet", vmin=0, vmax=4)
    ax.set_title("width = {}".format(width))
    # show histogram
    ax = axes[1, i]
    ax.hist(loss_barrier.flatten(), bins=100, range=(0, 8))
    # draw vertical line at 5, 50, 95 percentile
    for p in [5, 50, 95]:
        ax.axvline(np.percentile(loss_barrier, p), color="k", linestyle="--")
    ax.set_title("width = {}".format(width))
    ax.set_xlabel("loss barrier")
# show colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

# save figure
plt.savefig("naive_gaussian_loss_barrier_s512_d1.png", dpi=300)
plt.show()
"""

"""
# visualizing the interpolation loss

# load interpolation losses for width=3
int_losses = np.load(f"logs/sigmoid/gaussian/naive_interpolation_losses_s512_w3_d1.npy")

# create a 2*5 subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=True)

# for 10 random model pairs, show the interpolation loss
for i in range(10):
    # randomly select 2 models
    idx = np.random.choice(50, size=2, replace=False)
    # show interpolation loss
    ax = axes[i // 5, i % 5]
    ax.plot(int_losses[idx[0], idx[1]])
    ax.set_title("model {} vs model {}".format(idx[0], idx[1]))
    ax.set_xlabel("interpolation")
    ax.set_ylabel("interpolation loss")

# show
plt.show()
"""

"""
# visualizing model losses and accuracies

# create 5*4 subplots with enough space between them
fig, axes = plt.subplots(5, 4, figsize=(20, 20), squeeze=True)

# for widths [3, 4, 5, 6, 12], load the model losses and accuracies and show their histograms
for i, width in enumerate([3, 4, 5, 6, 12]):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    model_logs = np.load(f"logs/sigmoid/gaussian/logs_s512_w{width}_d1.npy")
    train_losses.append(model_logs[:, 0])
    test_losses.append(model_logs[:, 1])
    train_accuracies.append(model_logs[:, 2])
    test_accuracies.append(model_logs[:, 3])

    # show train loss histogram
    axes[i, 0].hist(train_losses, bins=10)
    if i == 0:
        axes[i, 0].set_title("width = {}".format(width))
        axes[i, 0].set_xlabel("train loss")
    # show test loss histogram
    axes[i, 1].hist(test_losses, bins=10)
    if i == 0:
        axes[i, 1].set_title("width = {}".format(width))
        axes[i, 1].set_xlabel("test loss")
    # show train accuracy histogram
    axes[i, 2].hist(train_accuracies, bins=10)
    if i == 0:
        axes[i, 2].set_title("width = {}".format(width))
        axes[i, 2].set_xlabel("train accuracy")
    # show test accuracy histogram
    axes[i, 3].hist(test_accuracies, bins=10)
    if i == 0:
        axes[i, 3].set_title("width = {}".format(width))
        axes[i, 3].set_xlabel("test accuracy")

# save
plt.savefig("gaussian_model_logs_s512_d1.png", dpi=300)
"""

"""
# building graphs using networkx

for width in [3, 4, 5, 6, 12]:
    # create a graph
    G = nx.Graph()

    # add 50 nodes
    for i in range(50):
        G.add_node(i)

    # load loss barrier
    loss_barrier = np.load(
        f"logs/sigmoid/gaussian/naive_loss_barriers_s512_w{width}_d1.npy"
    )

    # choose various thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    k_cliques_list = []
    max_cliques_list = []

    for threshold in thresholds:
        # add edges between nodes with loss barrier < threshold
        for i in range(50):
            for j in range(i + 1, 50):
                if loss_barrier[i, j] < threshold:
                    G.add_edge(i, j)

        # count k-cliques
        k_cliques = [50]
        k = 2
        count = 0

        while True:
            for clique in nx.find_cliques(G):
                count += comb(len(clique), k)
            if count == 0:
                break
            else:
                k_cliques.append(count)
                count = 0
                k += 1

        k_cliques_list.append(k_cliques)

        # count max cliques
        max_cliques = [0] * max([len(c) for c in nx.find_cliques(G)])

        for clique in nx.find_cliques(G):
            max_cliques[len(clique) - 1] += 1

        max_cliques_list.append(max_cliques)

    # pad k_cliques_list so that all lists have the same length
    max_len = max([len(l) for l in k_cliques_list])
    for i in range(len(k_cliques_list)):
        k_cliques_list[i] += [0] * (max_len - len(k_cliques_list[i]))

    # pad max_cliques_list so that all lists have the same length
    max_len = max([len(l) for l in max_cliques_list])
    for i in range(len(max_cliques_list)):
        max_cliques_list[i] += [0] * (max_len - len(max_cliques_list[i]))

    # transpose lists
    k_cliques_list = np.array(k_cliques_list).T
    max_cliques_list = np.array(max_cliques_list).T

    # normalize columns
    k_cliques_list_norm = k_cliques_list / k_cliques_list.sum(axis=0)
    max_cliques_list_norm = max_cliques_list / max_cliques_list.sum(axis=0)

    # create 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # in first subplot, show k-cliques as heatmap with annotations
    im = axes[0].imshow(
        k_cliques_list_norm, cmap="hot", interpolation="nearest", aspect="auto"
    )
    for i in range(len(k_cliques_list)):
        for j in range(len(k_cliques_list[0])):
            text = axes[0].text(
                j, i, k_cliques_list[i, j], ha="center", va="center", color="grey"
            )
    axes[0].set_xticks(np.arange(len(thresholds)))
    axes[0].set_xticklabels(thresholds)
    axes[0].set_yticks(np.arange(k_cliques_list.shape[0]))
    axes[0].set_yticklabels(range(1, k_cliques_list.shape[0] + 1))
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("k-clique size")
    axes[0].set_title("k-cliques")

    # in second subplot, show max cliques as heatmap with annotations
    im = axes[1].imshow(
        max_cliques_list_norm, cmap="hot", interpolation="nearest", aspect="auto"
    )
    for i in range(len(max_cliques_list)):
        for j in range(len(max_cliques_list[0])):
            text = axes[1].text(
                j, i, max_cliques_list[i, j], ha="center", va="center", color="grey"
            )
    axes[1].set_xticks(np.arange(len(thresholds)))
    axes[1].set_xticklabels(thresholds)
    axes[1].set_yticks(np.arange(max_cliques_list.shape[0]))
    axes[1].set_yticklabels(np.arange(1, max_cliques_list.shape[0] + 1))
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("max clique size")
    axes[1].set_title("max cliques")

    # save
    plt.savefig(f"naive_gaussian_graphs_s512_w{width}_d1.png", dpi=300)
"""

"""
# visualize interpolation losses

# load naive interpolation losses for width 5, 6, 12
naive_int_losses_5 = np.load(
    "logs/sigmoid/gaussian/naive_interpolation_losses_s512_w5_d1.npy"
)
naive_int_losses_6 = np.load(
    "logs/sigmoid/gaussian/naive_interpolation_losses_s512_w6_d1.npy"
)
naive_int_losses_12 = np.load(
    "logs/sigmoid/gaussian/naive_interpolation_losses_s512_w12_d1.npy"
)

# compute mean values (average across dim 0 and 1)
naive_int_losses_5_mean = naive_int_losses_5.mean(axis=(0, 1))
naive_int_losses_6_mean = naive_int_losses_6.mean(axis=(0, 1))
naive_int_losses_12_mean = naive_int_losses_12.mean(axis=(0, 1))

# compute standard deviations (average across dim 0 and 1)
naive_int_losses_5_std = naive_int_losses_5.std(axis=(0, 1))
naive_int_losses_6_std = naive_int_losses_6.std(axis=(0, 1))
naive_int_losses_12_std = naive_int_losses_12.std(axis=(0, 1))

# create 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

# set y as log scale
axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[2].set_yscale("log")

# int losses are stored as (m1, m2, step) - gather values at each step
naive_int_losses_5 = [
    list(naive_int_losses_5[:, :, i][np.triu_indices(50, 1)].flatten().squeeze())
    for i in range(11)
]

# plot violin in first subplot
axes[0].violinplot(naive_int_losses_5, np.arange(11), showmedians=True)
axes[0].plot(
    naive_int_losses_5_mean,
    color="red",
    linewidth=2,
    label="mean",
)
# show standard deviation as lines around mean
axes[0].plot(
    naive_int_losses_5_mean + naive_int_losses_5_std,
    color="red",
    linewidth=1,
    linestyle="--",
    label="std",
)
axes[0].plot(
    naive_int_losses_5_mean - naive_int_losses_5_std,
    color="red",
    linewidth=1,
    linestyle="--",
)
axes[0].set_xlabel("interpolation step")
axes[0].set_ylabel("interpolation loss")
# set y axis limits
axes[0].set_ylim(0, 2.0)
axes[0].set_title("width 5")
axes[0].legend()

# in second subplot, show naive interpolation losses for all pairs and mean in bold
naive_int_losses_6 = [
    list(naive_int_losses_6[:, :, i][np.triu_indices(50, 1)].flatten().squeeze())
    for i in range(11)
]
axes[1].violinplot(naive_int_losses_6, np.arange(11), showmedians=True)
axes[1].plot(
    naive_int_losses_6_mean,
    color="green",
    linewidth=2,
    label="mean",
)
axes[1].plot(
    naive_int_losses_6_mean + naive_int_losses_6_std,
    color="green",
    linewidth=1,
    linestyle="--",
    label="std",
)
axes[1].plot(
    naive_int_losses_6_mean - naive_int_losses_6_std,
    color="green",
    linewidth=1,
    linestyle="--",
)
axes[1].set_xlabel("interpolation step")
axes[1].set_ylabel("interpolation loss")
axes[1].set_ylim(0, 2.0)
axes[1].set_title("width 6")
axes[1].legend()

# in third subplot, show naive interpolation losses for all pairs and mean in bold
naive_int_losses_12 = [
    list(naive_int_losses_12[:, :, i][np.triu_indices(50, 1)].flatten().squeeze())
    for i in range(11)
]
axes[2].violinplot(naive_int_losses_12, np.arange(11), showmedians=True)
axes[2].plot(
    naive_int_losses_12_mean,
    color="blue",
    linewidth=2,
    label="mean",
)
axes[2].plot(
    naive_int_losses_12_mean + naive_int_losses_12_std,
    color="blue",
    linewidth=1,
    linestyle="--",
    label="std",
)
axes[2].plot(
    naive_int_losses_12_mean - naive_int_losses_12_std,
    color="blue",
    linewidth=1,
    linestyle="--",
)
axes[2].set_xlabel("interpolation step")
axes[2].set_ylabel("interpolation loss")
axes[2].set_ylim(0, 2.0)
axes[2].set_title("width 12")
axes[2].legend()

# save
plt.savefig("gaussian_interpolation_losses_vio.png", dpi=300)
"""

# visualize

# make 2*5 subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, width in enumerate([3, 4, 5, 6, 12]):
    naive_bar_losses = np.load(
        f"logs/sigmoid/gaussian/naive_max_interpolation_losses_s512_w{width}_d1.npy"
    )
    bar_losses = np.load(
        f"logs/sigmoid/gaussian/max_interpolation_losses_s512_w{width}_d1.npy"
    )

    # condense losses
    cond_naive_bar_losses = naive_bar_losses[np.triu_indices(50, 1)]
    cond_bar_losses = bar_losses[np.triu_indices(50, 1)]

    naive_link = linkage(cond_naive_bar_losses, method="ward")
    link = linkage(cond_bar_losses, method="ward")

    # reorder
    naive_bar_losses = naive_bar_losses[leaves_list(naive_link), :]
    naive_bar_losses = naive_bar_losses[:, leaves_list(naive_link)]
    bar_losses = bar_losses[leaves_list(link), :]
    bar_losses = bar_losses[:, leaves_list(link)]

    # plot
    axes[0, i].imshow(naive_bar_losses, cmap="viridis", vmin=0, vmax=0.1)
    axes[0, i].set_title(f"width {width}")
    axes[1, i].imshow(bar_losses, cmap="viridis", vmin=0, vmax=0.1)

    # set y axis labels
    if i == 0:
        axes[0, i].set_ylabel("naive")
        axes[1, i].set_ylabel("aligned")

# common colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(axes[0, 0].get_images()[0], cax=cbar_ax)

# save
plt.savefig("cluster_gaussian_max_losses_1.png", dpi=300)
