import torch
import numpy as np
import matplotlib.pyplot as plt
from architecture.MLP import FCNet
from utils import get_data, plotter, evaluate
from collections import OrderedDict
from matplotlib import cm, colorbar
from matplotlib.colors import Normalize, BoundaryNorm, LogNorm
from sklearn.cluster import SpectralClustering
import seaborn as sns
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve, PersistenceLandscape, PersistenceImage
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings

warnings.filterwarnings("ignore")

width = 512

# # load data from data/moons.npz
# file = np.load("data/moons.npz")
# X_train = file["X_train"]
# y_train = file["y_train"]
# X_test = file["X_test"]
# y_test = file["y_test"]

# # define train and test loaders
# train_loader = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(
#         torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
#     ),
#     batch_size=256,
#     shuffle=True,
# )
# test_loader = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(
#         torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
#     ),
#     batch_size=256,
#     shuffle=False,
# )

# choose two random indices from 0 to 49
idx1 = np.random.randint(0, 50)
idx2 = np.random.randint(0, 50)
while idx1 == idx2:
    idx2 = np.random.randint(0, 50)

# replace
idx1 = 33
idx2 = 39

# load both models
model1 = FCNet(2, width, 1, 1)
model2 = FCNet(2, width, 1, 1)

model1.load_state_dict(torch.load(f"models/moons/model_w{width}_{idx1}.pth"))
model2.load_state_dict(torch.load(f"models/moons/model_w{width}_{idx2}.pth"))

average_model = FCNet(2, width, 1, 1)
average_state_dict = OrderedDict()
for key in model1.state_dict():
    average_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2
average_model.load_state_dict(average_state_dict)

# config
widths = [8, 32, 128, 512]
num_models = 40
depth = 3
epochs = 50

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# # plotting neurons
# idx1 = 42
# idx2 = 16
# width = 512

# model1 = FCNet(2, width, 1, 1)
# model2 = FCNet(2, width, 1, 1)
# model1.load_state_dict(torch.load(f"models/moons/perm_model_w{width}_{idx1}.pth"))
# model2.load_state_dict(torch.load(f"models/moons/perm_model_w{width}_{idx2}.pth"))

# w_in1 = model1.layers[0].weight.detach().cpu().numpy()
# b_in1 = model1.layers[0].bias.detach().cpu().numpy()
# w_out1 = model1.layers[1].weight.detach().cpu().numpy()
# w_in2 = model2.layers[0].weight.detach().cpu().numpy()
# b_in2 = model2.layers[0].bias.detach().cpu().numpy()
# w_out2 = model2.layers[1].weight.detach().cpu().numpy()
# x1 = np.linspace(-3, 3, 100)
# x2 = np.linspace(-3, 3, 100)


# for j in range(width):
#     # plotting
#     fig, axs = plt.subplots(1, 2)

#     # set titles for each row
#     axs[0].set_title(f"model {idx1}")
#     axs[1].set_title(f"model {idx2}")
#     # x and y limits
#     axs[0].set_xlim(-3, 3)
#     axs[0].set_ylim(-3, 3)
#     axs[1].set_xlim(-3, 3)
#     axs[1].set_ylim(-3, 3)

#     axs[0].plot(
#         x1,
#         -(w_in1[j, 0] * x1 + b_in1[j]) / w_in1[j, 1],
#         linestyle="dashed",
#         linewidth=5,
#         alpha=np.sqrt(np.linalg.norm(w_in1[j]) ** 2 + b_in1[j] ** 2) / 10,
#         c="black",
#     )
#     axs[1].plot(
#         x1,
#         -(w_in2[j, 0] * x1 + b_in2[j]) / w_in2[j, 1],
#         linestyle="dashed",
#         linewidth=5,
#         alpha=np.sqrt(np.linalg.norm(w_in2[j]) ** 2 + b_in2[j] ** 2) / 10,
#         c="black",
#     )

#     # plot data
#     axs[0].scatter(
#         X_train[:50, 0], X_train[:50, 1], c=y_train[:50], cmap="coolwarm", s=4
#     )
#     axs[1].scatter(
#         X_train[:50, 0], X_train[:50, 1], c=y_train[:50], cmap="coolwarm", s=4
#     )

#     # color in the halfspaces based on the output weights
#     val = w_in1[j, 1] < 0
#     if val:
#         y_lim = 3
#     else:
#         y_lim = -3

#     if w_out1[:, j] > 0:
#         axs[0].fill_between(
#             x1,
#             -(w_in1[j, 0] * x1 + b_in1[j]) / w_in1[j, 1],
#             y_lim,
#             color="red",
#             alpha=np.abs(w_out1[:, j]) / 10,
#         )
#     else:
#         axs[0].fill_between(
#             x1,
#             -(w_in1[j, 0] * x1 + b_in1[j]) / w_in1[j, 1],
#             y_lim,
#             color="blue",
#             alpha=np.abs(w_out1[:, j]) / 10,
#         )

#     val = w_in2[j, 1] < 0
#     if val:
#         y_lim = 3
#     else:
#         y_lim = -3

#     if w_out2[:, j] > 0:
#         axs[1].fill_between(
#             x1,
#             -(w_in2[j, 0] * x1 + b_in2[j]) / w_in2[j, 1],
#             y_lim,
#             color="red",
#             alpha=np.abs(w_out2[:, j]) / 10,
#         )
#     else:
#         axs[1].fill_between(
#             x1,
#             -(w_in2[j, 0] * x1 + b_in2[j]) / w_in2[j, 1],
#             y_lim,
#             color="blue",
#             alpha=np.abs(w_out2[:, j]) / 10,
#         )

#     # aspec ratio
#     axs[0].set_aspect("equal")
#     axs[1].set_aspect("equal")

#     # save figure
#     fig.savefig(f"models_w{width}_{idx1}_{idx2}_{j}.png")
#     fig.clf()


# # given a cmap, divide it into n colors, and return the colors
# def get_colors(cmap, n):
#     colors = []
#     for i in range(n):
#         colors.append(cmap(i / n))
#     return colors


# # test
# colors = get_colors(plt.cm.coolwarm, 10)
# print(colors)


# # plot all three for all three models
# fig, axs = plt.subplots(1, 2)

# # set titles for each row
# axs[0].set_title(f"model {idx1}")
# axs[1].set_title(f"model {idx2}")

# for i, model in enumerate([model1, model2]):
#     # decision boundary
#     model.eval().to(device)
#     x1 = np.linspace(-3, 3, 100)
#     x2 = np.linspace(-3, 3, 100)
#     X1, X2 = np.meshgrid(x1, x2)
#     X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
#     X = torch.from_numpy(X).float().to(device)
#     y = model(X).detach().cpu().numpy().reshape(100, 100)
#     # apply sigmoid
#     y = 1 / (1 + np.exp(-y))

#     w_in = model.layers[0].weight.detach().cpu().numpy()
#     b_in = model.layers[0].bias.detach().cpu().numpy()
#     # plot the lines corresponding to each hidden node
#     for j in range(width):
#         axs[i].plot(
#             x1,
#             -(w_in[j, 0] * x1 + b_in[j]) / w_in[j, 1],
#             linestyle="dotted",
#             alpha=0.2,
#             c=f"C{j}",
#         )
#     axs[i].contourf(x1, x2, y, 3, cmap="RdBu_r", alpha=0.5)
#     # highlight line at 0.5
#     axs[i].contour(x1, x2, y, levels=[0.5], colors="k", linewidths=2)
#     axs[i].set_aspect("equal")
#     axs[i].set_xlim(-3, 3)
#     axs[i].set_ylim(-3, 3)

#     # plot the training data
#     for x_, y_ in train_loader:
#         x_ = x_.numpy()
#         y_ = y_.numpy()
#         axs[i].scatter(x_[y_ == 0, 0], x_[y_ == 0, 1], c="b", s=2)
#         axs[i].scatter(x_[y_ == 1, 0], x_[y_ == 1, 1], c="r", s=2)

# fig.suptitle(f"Decision boundary: width = {width}")
# plt.tight_layout()
# plt.savefig(f"moons_w{width}.png", dpi=600, bbox_inches="tight")

# # plot decision boundary for average model
# fig, axs = plt.subplots(1, 1)

# # decision boundary
# average_model.eval().to(device)
# x1 = np.linspace(-3, 3, 100)
# x2 = np.linspace(-3, 3, 100)
# X1, X2 = np.meshgrid(x1, x2)
# X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
# X = torch.from_numpy(X).float().to(device)
# y = average_model(X).detach().cpu().numpy().reshape(100, 100)
# # apply sigmoid
# y = 1 / (1 + np.exp(-y))

# w_in = average_model.layers[0].weight.detach().cpu().numpy()
# b_in = average_model.layers[0].bias.detach().cpu().numpy()
# # plot the lines corresponding to each hidden node
# for j in range(width):
#     axs.plot(
#         x1,
#         -(w_in[j, 0] * x1 + b_in[j]) / w_in[j, 1],
#         linestyle="dotted",
#         alpha=0.2,
#         c=f"C{j}",
#     )
# axs.contourf(x1, x2, y, 3, cmap="RdBu_r", alpha=0.5)
# # highlight line at 0.5
# axs.contour(x1, x2, y, levels=[0.5], colors="k", linewidths=2)
# axs.set_aspect("equal")
# axs.set_xlim(-3, 3)
# axs.set_ylim(-3, 3)

# # plot the training data
# for x_, y_ in train_loader:
#     x_ = x_.numpy()
#     y_ = y_.numpy()
#     axs.scatter(x_[y_ == 0, 0], x_[y_ == 0, 1], c="b", s=2)
#     axs.scatter(x_[y_ == 1, 0], x_[y_ == 1, 1], c="r", s=2)

# fig.suptitle(f"Average of models {idx1} and {idx2}")
# plt.tight_layout()
# plt.savefig(f"moons_w{width}_average.png", dpi=600, bbox_inches="tight")

# # plot the weights
# fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
# # size
# fig.set_size_inches(8, 4)

# # plot the weights
# for i, model in enumerate([model1, model2]):
#     model.eval().to(device)
#     w_in = model.layers[0].weight.detach().cpu().numpy()
#     b_in = model.layers[0].bias.detach().cpu().numpy()
#     w_out = model.layers[1].weight.detach().cpu().numpy()
#     # add bias to w_in
#     w_in = np.concatenate([w_in, b_in.reshape(-1, 1)], axis=1)
#     w_out_abs = np.abs(w_out) ** 2
#     w_in_norm = np.linalg.norm(w_in, axis=1) ** 2

#     w_strength = w_in_norm.squeeze() + w_out_abs.squeeze()
#     w_strength_rel = w_strength / np.sum(w_strength)
#     w_strength_rel = sorted(w_strength_rel, reverse=True)
#     # show cdf
#     axs[i].plot(np.cumsum(w_strength_rel), c=f"C{i}")
#     # y line at 0.95
#     axs[i].axhline(0.95, linestyle="dotted", c="red", alpha=0.5)
#     # x line corresponding to 0.95
#     axs[i].axvline(
#         np.argwhere(np.cumsum(w_strength_rel) > 0.95)[0][0],
#         linestyle="dotted",
#         c="red",
#         alpha=0.5,
#     )
#     # mark the point
#     axs[i].scatter(
#         np.argwhere(np.cumsum(w_strength_rel) > 0.95)[0][0],
#         0.95,
#         c="red",
#         alpha=0.5,
#     )
#     # text label
#     axs[i].text(
#         np.argwhere(np.cumsum(w_strength_rel) > 0.95)[0][0] + 2.0,
#         0.9,
#         f"({np.argwhere(np.cumsum(w_strength_rel) > 0.95)[0][0]}, 0.95)",
#         c="red",
#         alpha=0.5,
#     )

#     axs[i].set_title(f"model {idx1}" if i == 0 else f"model {idx2}")
#     # grid
#     axs[i].grid(True, alpha=0.5)
#     axs[i].set_xlabel("hidden nodes")

# fig.suptitle(f"Relative strength of hidden nodes")
# plt.tight_layout()
# plt.savefig(f"moons_w{width}_contrib.png", dpi=600, bbox_inches="tight")

# # plot weights
# fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
# # size
# fig.set_size_inches(8, 4)

# for i, model in enumerate([model1, model2]):
#     model.eval().to(device)
#     w_in = model.layers[0].weight.detach().cpu().numpy()
#     b_in = model.layers[0].bias.detach().cpu().numpy()
#     w_out = model.layers[1].weight.detach().cpu().numpy()
#     # add bias to w_in
#     w_in = np.concatenate([w_in, b_in.reshape(-1, 1)], axis=1)
#     w_in_norm = np.linalg.norm(w_in, axis=1) ** 2
#     w_out_abs = np.abs(w_out.squeeze()) ** 2

#     # plot w_in_norm vs w_out
#     axs[i].scatter(w_in_norm, w_out_abs, c="red", s=10, alpha=0.1)
#     axs[i].set_title(f"model {idx1}" if i == 0 else f"model {idx2}")
#     axs[i].set_xlabel("$\|w_{in}\|_2^2$")
#     axs[i].set_ylabel("$\|w_{out}\|^2$")
#     axs[i].set_xlim(-0.2, 5.0)
#     axs[i].set_ylim(-0.2, 2.0)
#     # add line at 0
#     axs[i].axhline(0, linestyle="dotted", c="grey", alpha=0.5)
#     # add vertical line at 0
#     axs[i].axvline(0, linestyle="dotted", c="grey", alpha=0.5)
#     # aspect ratio
#     axs[i].set_aspect("auto")

# fig.suptitle(f"Weight distribution")
# plt.tight_layout()
# plt.savefig(f"moons_w{width}_weights.png", dpi=600, bbox_inches="tight")


# given a model, return indices and fraction nodes whose relative contribution is less than 95%
def get_low_norm_nodes(model):
    w_in = model.layers[0].weight.detach().cpu().numpy()
    b_in = model.layers[0].bias.detach().cpu().numpy()
    w_out = model.layers[1].weight.detach().cpu().numpy()
    # add bias to w_in
    w_in = np.concatenate([w_in, b_in.reshape(-1, 1)], axis=1)
    w_out_abs = np.abs(w_out) ** 2
    w_in_norm = np.linalg.norm(w_in, axis=1) ** 2

    w_contrib = w_in_norm.squeeze() + w_out_abs.squeeze()
    w_contrib_rel = w_contrib / np.sum(w_contrib)
    # get sorted indices of nodes by relative contribution
    sorted_indices = np.argsort(w_contrib_rel)[::-1]
    # get sorted relative contributions in descending order
    sorted_contrib_rel = w_contrib_rel[sorted_indices]

    # get original indices of nodes whose cumulative relative contribution is more than 95%
    low_norm_indices = sorted_indices[np.cumsum(sorted_contrib_rel) > 0.95]
    # exclude the first node if list size is greater than 1
    if len(low_norm_indices) > 1:
        low_norm_indices = low_norm_indices[1:]
    else:
        low_norm_indices = []
    # get fraction of nodes whose cumulative relative contribution is more than 95%
    low_norm_fraction = len(low_norm_indices) / len(w_contrib_rel)
    # return indices and fraction
    return low_norm_indices, low_norm_fraction


# visualize perm interpolation losses
widths = [32]

epsilon = np.zeros((40, 40, len(widths)))
for data in ["test"]:
    for i, width in enumerate(widths):
        int_losses = np.load(f"logs/mnist/perm_cust_int_losses_{data}_w{width}.npy")
        for j in range(int_losses.shape[0]):
            for k in range(int_losses.shape[1]):
                if j == k:
                    continue
                if j > k:
                    epsilon[j, k, i] = epsilon[k, j, i]
                if j < k:
                    epsilon[j, k, i] = int_losses[j, k, :].max() - max(
                        int_losses[j, k, 0], int_losses[j, k, -1]
                    )

        g = sns.clustermap(
            epsilon[:, :, i],
            cmap="rocket",
            vmin=0,
            vmax=1.0,
            xticklabels=False,
            yticklabels=False,
            figsize=(8, 8),
            cbar_kws={"label": "$\epsilon$"},
            metric="euclidean",
            method="single",
        )
        # save the figure
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        # hide the colorbar
        g.cax.set_visible(False)
        # save the figure
        g.savefig(f"zoomed_perm_sim_w{width}.png", dpi=600, bbox_inches="tight")

# # plot the cosine similarity between incoming weights of node-node pairs
# # plot the weights
# for i, model in enumerate([model1, model2]):
#     model.eval().to(device)
#     w_in = model.layers[0].weight.detach().cpu().numpy()
#     b_in = model.layers[0].bias.detach().cpu().numpy()
#     w_out = model.layers[1].weight.detach().cpu().numpy()

#     # add bias as column to w_in
#     w_in = np.hstack((w_in, b_in.reshape(-1, 1)))
#     # get cosine similarity between incoming weights of node-node pairs
#     sim = (
#         w_in
#         @ w_in.T
#         / (
#             (
#                 np.linalg.norm(w_in, axis=1).reshape(-1, 1)
#                 @ np.linalg.norm(w_in, axis=1).reshape(1, -1)
#             )
#         )
#     )
#     D = 1 - sim
#     # 3D array required for persistence diagrams
#     D = np.array([D])

#     # Instantiate topological transformer
#     VR = VietorisRipsPersistence(
#         metric="precomputed", homology_dimensions=[0], n_jobs=-1
#     )

#     # Calculate persistence diagrams
#     diagrams = VR.fit_transform(D)

#     # Instantiate Betti curve transformer
#     Betti = BettiCurve()
#     curves = Betti.fit_transform(diagrams)

#     # Plot Betti curves
#     # set plotly params so y axis is log scale and grid is ON
#     fig = Betti.plot(
#         curves,
#         plotly_params={
#             "layout": {
#                 "yaxis_type": "log",
#                 "yaxis_gridcolor": "grey",
#                 "xaxis_gridcolor": "grey",
#                 # title
#                 "title": f"Clustering features",
#             }
#         },
#     )
#     # save the plotly figure
#     fig.write_image(f"moons_w512_betti_1.png", width=800, height=800)

#     # Plot persistence diagrams
#     fig1 = VR.plot(
#         diagrams, plotly_params={"layout": {"title": f"Persistence diagram"}}
#     )

#     # save the plotly figure
#     fig1.write_image(f"moons_w512_persistence_1.png", width=800, height=800)

# # plot the sim matrix
# g = sns.clustermap(
#     sim,
#     cmap="icefire",
#     vmin=-1.0,
#     vmax=1.0,
#     xticklabels=False,
#     yticklabels=False,
#     figsize=(16, 16),
#     cbar_kws={"label": "cosine similarity"},
#     metric="euclidean",
#     method="single",
# )
# # save the figure
# g.ax_row_dendrogram.set_visible(True)
# g.ax_col_dendrogram.set_visible(True)
# # hide the colorbar
# g.cax.set_visible(False)
# # save the figure
# g.savefig(f"moons_w{width}_sim_{i}.png", dpi=600, bbox_inches="tight")

# plot the epsilon matrix

# for i, width in enumerate(widths):
#     # Instantiate topological transformer
#     VR = VietorisRipsPersistence(
#         metric="precomputed", homology_dimensions=[0], n_jobs=-1
#     )

#     # Calculate persistence diagrams
#     diagrams = VR.fit_transform(np.array([epsilon[:, :, i]]))

#     # Instantiate Betti curve transformer
#     Betti = BettiCurve()
#     curves = Betti.fit_transform(diagrams)

#     # Plot Betti curves
#     # set plotly params so y axis is log scale and grid is ON
#     fig = Betti.plot(
#         curves,
#         plotly_params={
#             "layout": {
#                 "yaxis_type": "log",
#                 "yaxis_gridcolor": "grey",
#                 "xaxis_gridcolor": "grey",
#                 # title
#                 "title": f"Clustering models",
#             }
#         },
#     )
#     # save the plotly figure
#     fig.write_image(f"perm_sim_betti_w{width}.png", width=800, height=800)

#     # Plot persistence diagrams
#     fig1 = VR.plot(
#         diagrams, plotly_params={"layout": {"title": f"Persistence diagram"}}
#     )

#     # save the plotly figure
#     fig1.write_image(f"perm_sim_persistence_w{width}.png", width=800, height=800)

# g = sns.clustermap(
#     epsilon[:, :, i],
#     cmap="rocket",
#     vmin=0,
#     vmax=1.0,
#     xticklabels=False,
#     yticklabels=False,
#     figsize=(8, 8),
#     cbar_kws={"label": "$\epsilon$"},
#     metric="euclidean",
#     method="single",
# )
# # save the figure
# g.ax_row_dendrogram.set_visible(False)
# g.ax_col_dendrogram.set_visible(False)
# # hide the colorbar
# g.cax.set_visible(False)
# # save the figure
# g.savefig(f"naive_sim_w{width}.png", dpi=600, bbox_inches="tight")


def reduce_model(model, in_threshold=0.1, out_threshold=0.1, sim_threshold=0.99):
    # get the weights
    w_in = model.layers[0].weight.detach().cpu().numpy()
    b_in = model.layers[0].bias.detach().cpu().numpy()
    w_out = model.layers[1].weight.detach().cpu().numpy()
    b_out = model.layers[1].bias.detach().cpu().numpy()

    # remove nodes that share high node-node similarity
    # add bias as column to w_in
    w_in_vec = np.hstack((w_in, b_in.reshape(-1, 1)))

    # get cosine similarity between incoming weights of node-node pairs
    sim = (
        w_in_vec
        @ w_in_vec.T
        / (
            (
                np.linalg.norm(w_in_vec, axis=1).reshape(-1, 1)
                @ np.linalg.norm(w_in_vec, axis=1).reshape(1, -1)
            )
        )
    )
    # consider only upper triangular part and rest as -2
    sim = np.triu(sim, k=1) + np.tril(np.ones_like(sim) * -2, k=0)

    # keep track to nodes considered
    num_nodes_considered = 0

    # loop until all nodes are either kept or removed
    while num_nodes_considered < len(w_in):
        # get the highest similarity pair
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        # within [i, :], [:, i], find j with sim > sim_threshold
        j_0 = np.where(sim[i, :] > sim_threshold)[0]
        j_1 = np.where(sim[:, i] > sim_threshold)[0]
        # union of j_0, j_1
        j_indices = set(j_0).union(set(j_1))
        # keeping i, remove j, k
        w_in = np.delete(w_in, list(j_indices), axis=0)
        b_in = np.delete(b_in, list(j_indices), axis=0)
        # updating w_out
        v1 = w_out[:, i]
        v2 = w_out[:, list(j_indices)]
        n1 = w_in_vec[i, :]
        n2 = w_in_vec[list(j_indices), :]
        lamb = np.linalg.norm(v1) / np.linalg.norm(v2, axis=0)
        w_out[:, i] = v1 + np.sum(lamb.reshape(1, -1) * v2, axis=1)
        w_out = np.delete(w_out, list(j_indices), axis=1)
        # update w_in_vec
        w_in_vec = np.delete(w_in_vec, list(j_indices), axis=0)
        # update sim
        sim = np.delete(sim, list(j_indices), axis=0)
        sim = np.delete(sim, list(j_indices), axis=1)
        # update num_nodes_considered
        num_nodes_considered += len(j_indices) + 1

    # print(f"number of nodes: {len(w_in)}")

    # remove low w_out norm nodes
    low_norm_indices = set()
    # get the norm of w_out
    w_out_norm = np.linalg.norm(w_out, axis=0)
    # get the indices of nodes with norm < threshold
    low_norm_indices = low_norm_indices.union(
        set(np.where(w_out_norm < out_threshold)[0])
    )
    # get the norm of w_in_vec
    w_in_norm = np.linalg.norm(w_in_vec, axis=1)
    # get the indices of nodes with norm < threshold (intersection)
    low_norm_indices = low_norm_indices.intersection(
        set(np.where(w_in_norm < in_threshold)[0])
    )

    # remove the nodes
    w_in = np.delete(w_in, list(low_norm_indices), axis=0)
    b_in = np.delete(b_in, list(low_norm_indices), axis=0)
    w_out = np.delete(w_out, list(low_norm_indices), axis=1)
    # update w_in_vec
    w_in_vec = np.delete(w_in_vec, list(low_norm_indices), axis=0)

    # print(f"number of nodes: {len(w_in)}")

    # number of remaining nodes
    num_nodes = w_in.shape[0]

    # create new model
    reduced_model = FCNet(input_size=2, width=num_nodes, depth=1, output_size=1).to(
        device
    )

    # set weights
    reduced_model.layers[0].weight.data = torch.from_numpy(w_in).float().to(device)
    reduced_model.layers[0].bias.data = torch.from_numpy(b_in).float().to(device)
    reduced_model.layers[1].weight.data = torch.from_numpy(w_out).float().to(device)
    reduced_model.layers[1].bias.data = torch.from_numpy(b_out).float().to(device)

    return reduced_model, num_nodes


# model1.eval().to(device)
# model2.eval().to(device)

# # reduce the model
# reduced_model1, num_nodes1 = reduce_model(
#     model1, in_threshold=0.5, out_threshold=0.5, sim_threshold=0.95
# )
# reduced_model2, num_nodes2 = reduce_model(
#     model2, in_threshold=0.5, out_threshold=0.5, sim_threshold=0.95
# )

# # performance of reduced models
# l, a = evaluate(reduced_model1, test_loader)
# print(f"Reduced model 1: loss = {l:.4f}, accuracy = {a:.4f}")
# l, a = evaluate(reduced_model2, test_loader)
# print(f"Reduced model 2: loss = {l:.4f}, accuracy = {a:.4f}")

# # make average model of width = max(num_nodes1, num_nodes2)
# width = max(num_nodes1, num_nodes2)
# average_model = FCNet(2, width, 1, 1)
# average_state_dict = OrderedDict()
# for key in reduced_model1.state_dict():
#     # pad the smaller model with zeros
#     if key == "layers.0.weight":
#         # choose which model to pad
#         if num_nodes1 < num_nodes2:
#             pad = torch.zeros((num_nodes2 - num_nodes1, 2)).to(device)
#             average_state_dict[key] = (
#                 torch.concatenate((reduced_model1.state_dict()[key], pad), axis=0)
#                 + reduced_model2.state_dict()[key]
#             ) / 2
#         else:
#             pad = torch.zeros((num_nodes1 - num_nodes2, 2)).to(device)
#             average_state_dict[key] = (
#                 reduced_model1.state_dict()[key]
#                 + torch.concatenate((reduced_model2.state_dict()[key], pad), axis=0)
#             ) / 2
#     elif key == "layers.0.bias":
#         if num_nodes1 < num_nodes2:
#             pad = torch.zeros(num_nodes2 - num_nodes1).to(device)
#             average_state_dict[key] = (
#                 reduced_model2.state_dict()[key]
#                 + torch.concatenate((reduced_model1.state_dict()[key], pad), axis=0)
#             ) / 2
#         else:
#             pad = torch.zeros(num_nodes1 - num_nodes2).to(device)
#             average_state_dict[key] = (
#                 torch.concatenate((reduced_model2.state_dict()[key], pad), axis=0)
#                 + reduced_model1.state_dict()[key]
#             ) / 2
#     elif key == "layers.1.weight":
#         if num_nodes1 < num_nodes2:
#             pad = torch.zeros((1, num_nodes2 - num_nodes1)).to(device)
#             average_state_dict[key] = (
#                 torch.concatenate((reduced_model1.state_dict()[key], pad), axis=1)
#                 + reduced_model2.state_dict()[key]
#             ) / 2
#         else:
#             pad = torch.zeros((1, num_nodes1 - num_nodes2)).to(device)
#             average_state_dict[key] = (
#                 reduced_model1.state_dict()[key]
#                 + torch.concatenate((reduced_model2.state_dict()[key], pad), axis=1)
#             ) / 2
#     elif key == "layers.1.bias":
#         average_state_dict[key] = (
#             reduced_model1.state_dict()[key] + reduced_model2.state_dict()[key]
#         ) / 2
#     else:
#         # error handling
#         raise ValueError("key not found")
# average_model.load_state_dict(average_state_dict)

# # evaluate the average model
# average_model.eval().to(device)
# loss, acc = evaluate(average_model, test_loader)
# print(f"average model loss: {loss}, average model accuracy: {acc}")

# # node-node sim for reduced model1
# w_in1 = reduced_model1.layers[0].weight.data.cpu().numpy()
# b_in1 = reduced_model1.layers[0].bias.data.cpu().numpy()
# w_in1_vec = np.concatenate((w_in1, b_in1.reshape(-1, 1)), axis=1)

# # get cosine similarity between incoming weights of node-node pairs
# sim = (
#     w_in1_vec
#     @ w_in1_vec.T
#     / (
#         (
#             np.linalg.norm(w_in1_vec, axis=1).reshape(-1, 1)
#             @ np.linalg.norm(w_in1_vec, axis=1).reshape(1, -1)
#         )
#     )
# )

# # plot the cosine similarity matrix with seaborn clustermap
# g = sns.clustermap(
#     sim,
#     cmap="vlag",
#     vmin=-1,
#     vmax=1,
#     xticklabels=False,
#     yticklabels=False,
#     figsize=(8, 8),
#     cbar_kws={"label": "cosine similarity"},
#     metric="euclidean",
#     method="single",
# )
# # save the figure
# g.ax_row_dendrogram.set_visible(False)
# g.ax_col_dendrogram.set_visible(False)
# # hide the colorbar
# g.cax.set_visible(False)
# # save the figure
# g.savefig(f"moons_sim.png", dpi=600, bbox_inches="tight")


# # Lets see if SWA averages are equivalent naive or aligned averaging
# # SWA
# from torch.optim.swa_utils import AveragedModel, SWALR

# # Lets keep model 0 as ref
# widths = [4, 8, 16, 32, 128, 512]
# model_losses = np.zeros((len(widths), 21))
# swa_model_losses = np.zeros((len(widths), 21))

# for i, width in enumerate(widths):
#     model = FCNet(2, width, 1, 1).to(device)
#     model.load_state_dict(torch.load(f"models/moons/model_w{width}_0.pth"))

#     # model
#     loss, _ = evaluate(model, test_loader)
#     model_losses[i, 0] = loss

#     criterion = torch.nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
#     swa_model = AveragedModel(model).to(device).train()
#     swa_scheduler = SWALR(optimizer, swa_lr=0.05)

#     # as the model is already trained, we start swa right away
#     swa_start = 0
#     swa_model.update_parameters(model)
#     swa_scheduler.step()

#     # SWA
#     swa_model.eval()
#     loss, _ = evaluate(swa_model, test_loader)
#     swa_model_losses[i, 0] = loss
#     swa_model.train()

#     for epoch in range(20):
#         for x, y in train_loader:
#             # model has 1 output
#             y = y.unsqueeze(1)
#             # Forward pass
#             optimizer.zero_grad()
#             y_pred = model(x.to(device))
#             loss = criterion(y_pred, y.to(device))
#             # Backward pass
#             loss.backward()
#             optimizer.step()
#         # save the model every epoch
#         torch.save(model.state_dict(), f"models/moons/swain_w{width}_{epoch}.pth")
#         # evaluate the model
#         loss, _ = evaluate(model, test_loader)
#         model_losses[i, epoch + 1] = loss
#         # update the swa model
#         if epoch >= swa_start:
#             swa_model.update_parameters(model)
#             swa_scheduler.step()
#         # save the model every epoch
#         torch.save(swa_model.state_dict(), f"models/moons/swa_w{width}_{epoch}.pth")
#         # evaluate the swa model
#         swa_model.eval()
#         loss, _ = evaluate(swa_model, test_loader)
#         swa_model_losses[i, epoch + 1] = loss
#         swa_model.train()

# # save the losses
# np.save("logs/moons/swain_model_losses.npy", model_losses)
# np.save("logs/moons/swa_model_losses.npy", swa_model_losses)

# # plot the losses
# model_losses = np.load("logs/moons/swain_model_losses.npy")
# swa_model_losses = np.load("logs/moons/swa_model_losses.npy")

# fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
# # y in log scale
# axes[0].set_yscale("log")
# axes[1].set_yscale("log")
# for i, width in enumerate(widths):
#     axes[i % 2].plot(
#         model_losses[i],
#         linestyle="--",
#         marker="o",
#         color=f"C{i}",
#         markersize=4,
#         alpha=0.5,
#     )
#     axes[i % 2].plot(
#         swa_model_losses[i],
#         label=f"width {width}",
#         marker="o",
#         color=f"C{i}",
#         markersize=4,
#         alpha=0.5,
#     )
#     axes[i % 2].set_xlabel("Epochs")
#     axes[i % 2].set_ylabel("Test loss")
# # common legend
# # get legent from 0 and 1 axes
# handles, labels = axes[0].get_legend_handles_labels()
# # append the legend from the second axes
# handles += axes[1].get_legend_handles_labels()[0]
# labels += axes[1].get_legend_handles_labels()[1]
# # add the legend
# axes[1].legend(handles, labels, loc="upper right")
# # title
# axes[0].set_title("SWA vs. model loss")
# # save the figure
# fig.savefig("swa_losses.png", dpi=600, bbox_inches="tight")
