# %% [markdown]
# Lets start simple: 1 hidden layer networks trained on moons dataset

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage

from architecture.MLP import FCNet, FCNet_multiclass, train, train_multiclass
from permute import permute_align
from utils import (
    evaluate,
    evaluate_multiclass,
    interpolation_losses,
    loss_barrier,
)
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms

# %%
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
widths = [4, 8, 16, 32, 128, 512]
num_models = 1
depth = 3
epochs = 100

# # load data from data/moons.npz
# file = np.load("data/moons.npz")
# X_train = file["X_train"]
# y_train = file["y_train"]
# X_test = file["X_test"]
# y_test = file["y_test"]

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=64,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=64,
    shuffle=True,
)

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


for width in widths:
    # data structure to store losses and accuracies
    logs = np.zeros((num_models, 4))

    # Define and train many models
    for i in range(num_models):
        model = FCNet_multiclass(
            input_size=784, width=width, depth=depth, output_size=10
        )
        train_multiclass(
            model,
            train_loader,
            epochs=epochs,
            lr=0.1,
            model_name=f"mnist/model_w{width}_{i}",
        )

        # evaluate
        model.eval()

        train_loss, train_acc = evaluate_multiclass(model, train_loader)
        test_loss, test_acc = evaluate_multiclass(model, test_loader)

        logs[i, 0] = train_loss
        logs[i, 1] = test_loss
        logs[i, 2] = train_acc
        logs[i, 3] = test_acc

    # save the logs
    np.save(f"logs/mnist/logs_w{width}", logs)

# # visualizing model losses and accuracies
# fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
# # title
# fig.suptitle("Full and reduced models of width 512")

# # for widths, load the model losses and accuracies and show their stacked histograms
# train_losses = np.zeros((num_models, len(widths)))
# test_losses = np.zeros((num_models, len(widths)))
# train_accs = np.zeros((num_models, len(widths)))
# test_accs = np.zeros((num_models, len(widths)))

# red_train_losses = np.zeros((num_models, len(widths)))
# red_test_losses = np.zeros((num_models, len(widths)))
# red_train_accs = np.zeros((num_models, len(widths)))
# red_test_accs = np.zeros((num_models, len(widths)))
# red_num_nodes = np.zeros((num_models, len(widths)))

# for i, width in enumerate([512]):
#     model_logs = np.load(f"logs/moons/logs_w{width}.npy")
#     red_model_logs = np.load(f"logs/moons/logs_red_w{width}.npy")
#     train_losses[:, i] = model_logs[:, 0]
#     test_losses[:, i] = model_logs[:, 1]
#     train_accs[:, i] = model_logs[:, 2]
#     test_accs[:, i] = model_logs[:, 3]

#     red_train_losses[:, i] = red_model_logs[:, 0]
#     red_test_losses[:, i] = red_model_logs[:, 1]
#     red_train_accs[:, i] = red_model_logs[:, 2]
#     red_test_accs[:, i] = red_model_logs[:, 3]
#     red_num_nodes[:, i] = red_model_logs[:, 4]

# # show train loss vs train acc
# # set log scale
# axes[0].set_xscale("log")
# axes[0].set_yscale("log")
# axes[0].scatter(
#     train_losses[:, 0],
#     train_accs[:, 0],
#     label="full",
#     color="blue",
#     alpha=0.5,
#     s=2,
# )
# axes[0].scatter(
#     red_train_losses[:, 0],
#     red_train_accs[:, 0],
#     label="reduced",
#     color="red",
#     alpha=0.5,
#     s=2,
# )
# axes[0].set_xlabel("train loss")
# axes[0].set_ylabel("train accuracy")
# axes[0].legend()

# # show test loss vs test acc
# # set log scale
# axes[1].set_xscale("log")
# axes[1].set_yscale("log")
# axes[1].scatter(
#     test_losses[:, 0],
#     test_accs[:, 0],
#     label="full",
#     color="blue",
#     alpha=0.5,
#     s=2,
# )
# axes[1].scatter(
#     red_test_losses[:, 0],
#     red_test_accs[:, 0],
#     label="reduced",
#     color="red",
#     alpha=0.5,
#     s=2,
# )
# axes[1].set_xlabel("test loss")
# axes[1].set_ylabel("test accuracy")
# axes[1].legend()

# # padding
# fig.tight_layout(pad=1.0)
# # save
# plt.savefig("model_comparison.png", dpi=600, bbox_inches="tight")
# plt.close()

# # plot hist plots of number of nodes
# fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharey=True, sharex=True)
# # title
# fig.suptitle("Number of nodes in reduced model width 512")
# # plot
# axes.hist(red_num_nodes[:, 0], bins=10, color="red", alpha=0.5)
# axes.set_xlabel("number of nodes")
# axes.set_ylabel("count")
# # padding
# fig.tight_layout(pad=1.0)
# # save
# plt.savefig("num_nodes.png", dpi=600, bbox_inches="tight")
# plt.close()

# for width in widths:
#     # data structure to store losses and accuracies
#     logs = np.zeros((num_models, 4))

#     # Define and train many models
#     models = []
#     for i in range(num_models):
#         model = FCNet(input_size=2, width=width, depth=depth, output_size=1)
#         train(
#             model,
#             train_loader,
#             epochs=epochs,
#             lr=0.1,
#             model_name=f"moons/model_w{width}_{i}",
#         )
#         models.append(model)

#         # evaluate
#         model.eval()

#         train_loss, train_acc = evaluate(model, train_loader)
#         test_loss, test_acc = evaluate(model, test_loader)

#         logs[i, 0] = train_loss
#         logs[i, 1] = test_loss
#         logs[i, 2] = train_acc
#         logs[i, 3] = test_acc

#         # save the logs
#         np.save(f"logs/moons/logs_w{width}", logs)

# # visualizing model losses and accuracies
# fig, axes = plt.subplots(2, 2, figsize=(5, 7), sharey=True)
# # title
# fig.suptitle("2-layer MLPs on moons")

# # for widths, load the model losses and accuracies and show their stacked histograms
# train_losses = np.zeros((num_models, len(widths)))
# test_losses = np.zeros((num_models, len(widths)))
# train_accs = np.zeros((num_models, len(widths)))
# test_accs = np.zeros((num_models, len(widths)))

# for i, width in enumerate(widths):
#     model_logs = np.load(
#         f"logs/moons/logs_w{width}.npy"
#     )
#     train_losses[:, i] = model_logs[:, 0]
#     test_losses[:, i] = model_logs[:, 1]
#     train_accs[:, i] = model_logs[:, 2]
#     test_accs[:, i] = model_logs[:, 3]

# # show train loss
# axes[0, 0].hist(train_losses, bins=10, stacked=True, label=widths, range=(0, 0.3), alpha=0.8)
# axes[0, 0].set_title("Train loss")

# axes[0, 1].hist(test_losses, bins=10, stacked=True, label=widths, range=(0, 0.3), alpha=0.8)
# axes[0, 1].set_title("Test loss")
# axes[0, 1].legend(title="widths")
# axes[1, 0].hist(train_accs, bins=10, stacked=True, label=widths, range=(0.85, 1), alpha=0.8)
# axes[1, 0].set_title("Train accuracy")

# axes[1, 1].hist(test_accs, bins=10, stacked=True, label=widths, range=(0.85, 1), alpha=0.8)
# axes[1, 1].set_title("Test accuracy")


# # padding
# fig.tight_layout(pad=1.0)
# # save
# plt.savefig("model_performance.png", dpi=300)


# given ref model and model, return realigned model
def weight_matching(ref_model, model):
    width = ref_model.layers[0].weight.shape[0]
    # compute cost
    cost = torch.zeros((width, width)).to(device)
    cost += torch.matmul(ref_model.layers[0].weight, model.layers[0].weight.T)
    cost += torch.matmul(
        ref_model.layers[0].bias.unsqueeze(1), model.layers[0].bias.unsqueeze(0)
    )
    cost += torch.matmul(ref_model.layers[1].weight.T, model.layers[1].weight)

    # get permutation using hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy(), maximize=True)
    perm = torch.zeros(cost.shape).to(device)
    perm[row_ind, col_ind] = 1

    # realign model
    model.layers[0].weight = torch.nn.Parameter(
        torch.matmul(perm, model.layers[0].weight)
    )
    model.layers[0].bias = torch.nn.Parameter(
        torch.matmul(perm, model.layers[0].bias.unsqueeze(1)).squeeze()
    )
    model.layers[1].weight = torch.nn.Parameter(
        torch.matmul(model.layers[1].weight, perm.T)
    )

    return model


# for w in widths:
#     ref_model = FCNet(2, w, 1, 1).to(device)
#     ref_model.load_state_dict(torch.load(f"models/moons/model_w{w}_0.pth"))
#     ref_model.eval()

#     for i in range(1, num_models):
#         model = FCNet(2, w, 1, 1).to(device)
#         model.load_state_dict(torch.load(f"models/moons/model_w{w}_{i}.pth"))
#         model.eval()

#         model, _ = permute_align(
#             model,
#             ref_model,
#             test_loader,
#             epochs=100,
#             device=device,
#         )

#         torch.save(model.state_dict(), f"models/moons/perm_cust_model_w{w}_{i}.pth")

# for width in widths:
#     models = []
#     ref_model = FCNet(2, width, 1, 1).to(device)
#     ref_model.load_state_dict(torch.load(f"models/moons/model_w{width}_0.pth"))
#     models.append(ref_model)
#     for i in range(1, num_models):
#         model = FCNet(2, width, 1, 1).to(device)
#         model.load_state_dict(
#             torch.load(f"models/moons/perm_cust_model_w{width}_{i}.pth")
#         )
#         models.append(model)

#     for data in ["train", "test"]:
#         # choose loader
#         if data == "train":
#             loader = train_loader
#         else:
#             loader = test_loader

#         # data structure to store interpolation losses
#         int_losses = np.zeros((num_models, num_models, 11))

#         # data structure to store loss barriers
#         barriers = np.zeros((num_models, num_models))

#         # data structure to store max barrier
#         max_barriers = np.zeros((num_models, num_models))

#         # compute interpolation loss for each pair of models
#         # log the results
#         for i in range(num_models):
#             for j in range(num_models):
#                 if i == j:
#                     continue
#                 if i > j:
#                     int_losses[i, j, :] = int_losses[j, i, :]
#                     barriers[i, j] = barriers[j, i]
#                     max_barriers[i, j] = max_barriers[j, i]
#                     continue
#                 if i < j:
#                     int_losses[i, j, :] = interpolation_losses(
#                         models[i], models[j], loader
#                     )
#                     barriers[i, j] = loss_barrier(int_losses[i, j, :])
#                     max_barriers[i, j] = max(int_losses[i, j, :])

#         np.save(
#             f"logs/moons/perm_cust_int_losses_{data}_w{width}",
#             int_losses,
#         )
#         np.save(
#             f"logs/moons/perm_cust_barriers_{data}_w{width}",
#             barriers,
#         )
#         np.save(
#             f"logs/moons/perm_cust_max_barriers_{data}_w{width}",
#             max_barriers,
#         )

# # visualize naive interpolation losses
# for data in ["train", "test"]:
#     # create 3*2 subplots
#     fig, axes = plt.subplots(3, 2, figsize=(7, 8), sharex=True, sharey=True)

#     for i, width in enumerate(widths):
#         int_losses = np.load(f"logs/moons/perm_cust_int_losses_{data}_w{width}.npy")

#         # compute mean values (average across dim 0 and 1)
#         int_losses_mean = int_losses.mean(axis=(0, 1))

#         # compute standard deviations (average across dim 0 and 1)
#         int_losses_std = int_losses.std(axis=(0, 1))

#         # plot individual losses as lines in same subplot
#         for j in range(int_losses.shape[0]):
#             for k in range(int_losses.shape[1]):
#                 axes[i // 2, i % 2].plot(
#                     int_losses[j, k], linewidth=0.5, color="grey", alpha=0.1
#                 )
#         # plot mean as line in same subplot
#         axes[i // 2, i % 2].plot(
#             int_losses_mean, color="red", linewidth=2, label="mean"
#         )
#         # show standard deviation as lines around mean
#         axes[i // 2, i % 2].plot(
#             int_losses_mean + int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         axes[i // 2, i % 2].plot(
#             int_losses_mean - int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         # set x axis label
#         axes[i // 2, i % 2].set_xlabel("$\\alpha$")
#         # set x axis ticks (0 to 1 in steps of 0.2)
#         axes[i // 2, i % 2].set_xticks(np.arange(0, 11, 2))
#         # set x axis tick labels (0 to 1 in steps of 0.1)
#         axes[i // 2, i % 2].set_xticklabels([f"{i / 10:.1f}" for i in range(0, 11, 2)])
#         # set x axis limits
#         axes[i // 2, i % 2].set_xlim(0, 10)
#         # set y axis label
#         axes[i // 2, i % 2].set_ylabel("loss")
#         # set y axis limits
#         axes[i // 2, i % 2].set_ylim(0, 1)
#         # set title
#         axes[i // 2, i % 2].set_title(f"width {width}")
#         # grid
#         axes[i // 2, i % 2].grid()

#     # set legend
#     axes[0, 0].legend(loc="upper right")

#     # set suptitle
#     fig.suptitle(f"Interpolation between permuted weights: {data} loss")
#     # tight layout
#     fig.tight_layout()
#     # save
#     plt.savefig(f"perm_cust_interpolation_losses_{data}.png", dpi=600)

# # visualize perm interpolation losses
# epsilon = np.zeros((int((50 * 49) / 2), 6))
# for data in ["test"]:
#     for i, width in enumerate(widths):
#         int_losses = np.load(f"logs/moons/perm_cust_int_losses_{data}_w{width}.npy")
#         idx = 0
#         for j in range(int_losses.shape[0]):
#             for k in range(int_losses.shape[1]):
#                 if j == k:
#                     continue
#                 if j > k:
#                     continue
#                 if j < k:
#                     epsilon[idx, i] = int_losses[j, k, :].max() - max(
#                         int_losses[j, k, 0], int_losses[j, k, -1]
#                     )
#                     idx += 1
#                 else:
#                     continue

# # visualize
# fig, ax = plt.subplots(1, 1, figsize=(7, 4))
# # x axis in log scale
# ax.set_xscale("log")
# # plot epsilon as violin plot
# ax.violinplot(
#     epsilon,
#     widths,
#     showmeans=True,
#     showextrema=False,
#     widths=np.array(widths) / 5,
#     showmedians=True,
# )
# # plot mean as line
# ax.plot(
#     widths,
#     epsilon.mean(axis=0),
#     color="blue",
#     marker="o",
#     markersize=2,
#     label="mean",
#     linestyle="dashed",
# )
# # plot median as line
# ax.plot(
#     widths,
#     np.median(epsilon, axis=0),
#     color="blue",
#     marker="o",
#     markersize=2,
#     label="median",
#     linestyle="dotted",
# )
# # set x axis label
# ax.set_xlabel("width")
# # set x axis ticks
# ax.set_xticks(widths)
# # set x axis tick labels
# ax.set_xticklabels(widths)
# # set x axis limits
# ax.set_xlim(3.5, 580)
# # set y axis limits
# ax.set_ylim(0, 0.2)
# # set y axis label
# ax.set_ylabel("$\\epsilon$")
# # grid
# ax.grid()
# # set legend
# ax.legend(loc="upper right")
# # set suptitle
# fig.suptitle("$\\epsilon$-linear mode connectivity in SWA samples")
# # tight layout
# # fig.tight_layout()
# # save
# plt.savefig(f"perm_cust_epsilon.png", dpi=600)

# # visualize naive interpolation losses zoomed in
# for data in ["train", "test"]:
#     # create 3*2 subplots
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

#     for i, width in enumerate([128, 512]):
#         int_losses = np.load(f"logs/moons/perm_cust_int_losses_{data}_w{width}.npy")

#         # compute mean values (average across dim 0 and 1)
#         int_losses_mean = int_losses.mean(axis=(0, 1))

#         # compute standard deviations (average across dim 0 and 1)
#         int_losses_std = int_losses.std(axis=(0, 1))

#         # plot individual losses as lines in same subplot
#         for j in range(int_losses.shape[0]):
#             for k in range(int_losses.shape[1]):
#                 axes[i].plot(int_losses[j, k], linewidth=0.5, color="grey", alpha=0.1)
#         # plot mean as line in same subplot
#         axes[i].plot(int_losses_mean, color="red", linewidth=2, label="mean")
#         # show standard deviation as lines around mean
#         axes[i].plot(
#             int_losses_mean + int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         axes[i].plot(
#             int_losses_mean - int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         # set x axis label
#         axes[i].set_xlabel("$\\alpha$")
#         # set x axis ticks (0 to 1 in steps of 0.2)
#         axes[i].set_xticks(np.arange(0, 11, 2))
#         # set x axis tick labels (0 to 1 in steps of 0.1)
#         axes[i].set_xticklabels([f"{i / 10:.1f}" for i in range(0, 11, 2)])
#         # set x axis limits
#         axes[i].set_xlim(0, 10)
#         # set y axis label
#         axes[i].set_ylabel("loss")
#         # set y axis limits
#         axes[i].set_ylim(0, 0.02)
#         # set title
#         axes[i].set_title(f"width {width}")
#         # grid
#         axes[i].grid()

#     # set legend
#     axes[0].legend(loc="upper right")

#     # set suptitle
#     fig.suptitle(f"Interpolation between permuted weights (zoomed): {data} loss")
#     # tight layout
#     fig.tight_layout()
#     # save
#     plt.savefig(f"zoomed_perm_cust_interpolation_losses_{data}.png", dpi=600)


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


# frac_nodes = np.zeros((50, 6))
# for j, width in enumerate(widths):
#     # load model
#     for i in range(50):
#         model = FCNet(input_size=2, width=width, depth=1, output_size=1).to(device)
#         model.load_state_dict(torch.load(f"models/moons/model_w{width}_{i}.pth"))
#         # get fraction of nodes whose relative contribution is less than 95%
#         _, frac_nodes[i, j] = get_low_norm_nodes(model)

# # visualize
# fig, ax = plt.subplots(1, 1, figsize=(7, 4))
# # x axis in log scale
# ax.set_xscale("log")
# # plot fraction as violin plot
# ax.violinplot(frac_nodes, widths, showmeans=True, widths=np.array(widths) / 5)
# # line plot of mean fraction
# ax.plot(
#     widths,
#     frac_nodes.mean(axis=0),
#     color="blue",
#     marker="o",
#     markersize=2,
#     label="mean",
#     linestyle="dashed",
# )
# # set x axis label
# ax.set_xlabel("width")
# # set x axis ticks
# ax.set_xticks(widths)
# # set x axis tick labels
# ax.set_xticklabels(widths)
# # set x axis limits
# ax.set_xlim(3.5, 580)
# # set y axis label
# ax.set_ylabel("fraction of nodes")
# # legend
# ax.legend(loc="upper right")
# # grid
# ax.grid()
# # set suptitle
# fig.suptitle("Hidden node fraction with low relative strength")
# # tight layout
# # fig.tight_layout()
# # save
# fig.savefig(f"low_norm_fraction.png", dpi=600, bbox_inches="tight")


def pad_models(model1, model2):
    # pad the smaller model with nodes with zero weights
    width1 = len(model1.layers[0].weight)
    width2 = len(model2.layers[0].weight)
    if width1 > width2:
        # pad model2
        pad = width1 - width2
        model2.layers[0].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model2.layers[0].weight,
                    torch.zeros(pad, model2.layers[0].weight.shape[1]).to(device),
                ],
                dim=0,
            )
        )
        model2.layers[1].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model2.layers[1].weight,
                    torch.zeros(model2.layers[1].weight.shape[0], pad).to(device),
                ],
                dim=1,
            )
        )
        model2.layers[0].bias = torch.nn.Parameter(
            torch.cat([model2.layers[0].bias, torch.zeros(pad).to(device)], dim=0)
        )
    elif width1 < width2:
        # pad model1
        pad = width2 - width1
        model1.layers[0].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model1.layers[0].weight,
                    torch.zeros(pad, model1.layers[0].weight.shape[1]).to(device),
                ],
                dim=0,
            )
        )
        model1.layers[1].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model1.layers[1].weight,
                    torch.zeros(model1.layers[1].weight.shape[0], pad).to(device),
                ],
                dim=1,
            )
        )
        model1.layers[0].bias = torch.nn.Parameter(
            torch.cat([model1.layers[0].bias, torch.zeros(pad).to(device)], dim=0)
        )
    return model1, model2


# widths = [4, 8, 16, 32, 128, 512]
# num_models = 21

# for width in widths:
#     models = []
#     model = FCNet(input_size=2, width=width, depth=1, output_size=1).to(device)
#     model.load_state_dict(torch.load(f"models/moons/model_w{width}_0.pth"))
#     models.append(model)
#     for i in range(num_models - 1):
#         model = FCNet(input_size=2, width=width, depth=1, output_size=1).to(device)
#         model.load_state_dict(torch.load(f"models/moons/swain_w{width}_{i}.pth"))
#         models.append(model)

#     for data in ["train", "test"]:
#         # choose loader
#         if data == "train":
#             loader = train_loader
#         else:
#             loader = test_loader

#         # data structure to store interpolation losses
#         int_losses = np.zeros((num_models, num_models, 11))

#         # data structure to store loss barriers
#         barriers = np.zeros((num_models, num_models))

#         # data structure to store max barrier
#         max_barriers = np.zeros((num_models, num_models))

#         # compute interpolation loss for each pair of models
#         # log the results
#         for i in range(num_models):
#             for j in range(num_models):
#                 if i == j:
#                     continue
#                 if i > j:
#                     int_losses[i, j, :] = int_losses[j, i, :]
#                     barriers[i, j] = barriers[j, i]
#                     max_barriers[i, j] = max_barriers[j, i]
#                     continue
#                 if i < j:
#                     int_losses[i, j, :] = interpolation_losses(
#                         models[i], models[j], loader
#                     )
#                     barriers[i, j] = loss_barrier(int_losses[i, j, :])
#                     max_barriers[i, j] = max(int_losses[i, j, :])

#         np.save(
#             f"logs/moons/naive_int_losses_{data}_swa_w{width}",
#             int_losses,
#         )
#         np.save(
#             f"logs/moons/naive_barriers_{data}_swa_w{width}",
#             barriers,
#         )
#         np.save(
#             f"logs/moons/naive_max_barriers_{data}_swa_w{width}",
#             max_barriers,
#         )

# # visualize naive interpolation losses
# for data in ["train", "test"]:
#     # create 3*2 subplots
#     fig, axes = plt.subplots(3, 2, figsize=(7, 8), sharex=True)

#     for i, width in enumerate(widths):
#         int_losses = np.load(f"logs/moons/naive_int_losses_{data}_swa_w{width}.npy")

#         # compute mean values (average across dim 0 and 1)
#         int_losses_mean = int_losses.mean(axis=(0, 1))

#         # compute standard deviations (average across dim 0 and 1)
#         int_losses_std = int_losses.std(axis=(0, 1))

#         # plot individual losses as lines in same subplot
#         for j in range(int_losses.shape[0]):
#             for k in range(int_losses.shape[1]):
#                 axes[i // 2, i % 2].plot(
#                     int_losses[j, k], linewidth=0.5, color="grey", alpha=0.1
#                 )
#         # plot mean as line in same subplot
#         axes[i // 2, i % 2].plot(
#             int_losses_mean, color="red", linewidth=2, label="mean"
#         )
#         # show standard deviation as lines around mean
#         axes[i // 2, i % 2].plot(
#             int_losses_mean + int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         axes[i // 2, i % 2].plot(
#             int_losses_mean - int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         # set x axis label
#         axes[i // 2, i % 2].set_xlabel("$\\alpha$")
#         # set x axis ticks (0 to 1 in steps of 0.2)
#         axes[i // 2, i % 2].set_xticks(np.arange(0, 11, 2))
#         # set x axis tick labels (0 to 1 in steps of 0.1)
#         axes[i // 2, i % 2].set_xticklabels([f"{i / 10:.1f}" for i in range(0, 11, 2)])
#         # set x axis limits
#         axes[i // 2, i % 2].set_xlim(0, 10)
#         # set y axis label
#         axes[i // 2, i % 2].set_ylabel("loss")
#         # set title
#         axes[i // 2, i % 2].set_title(f"width {width}")
#         # grid
#         axes[i // 2, i % 2].grid()

#     # set legend
#     axes[0, 0].legend(loc="upper right")

#     # set suptitle
#     fig.suptitle(f"Interpolation between SWA samples: {data} loss")
#     # tight layout
#     fig.tight_layout()
#     # save
#     plt.savefig(f"swa_interpolation_losses_{data}.png", dpi=600)

# # visualize naive interpolation losses
# # create 3*2 subplots
# fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
# for i, data in enumerate(["train", "test"]):
#     for width in [512]:
#         int_losses = np.load(f"logs/moons/naive_int_losses_{data}_swa_w{width}.npy")

#         # compute mean values (average across dim 0 and 1)
#         int_losses_mean = int_losses.mean(axis=(0, 1))

#         # compute standard deviations (average across dim 0 and 1)
#         int_losses_std = int_losses.std(axis=(0, 1))

#         # plot individual losses as lines in same subplot
#         for j in range(int_losses.shape[0]):
#             for k in range(int_losses.shape[1]):
#                 axes[i].plot(int_losses[j, k], color="grey", linewidth=0.5, alpha=0.2)
#         # plot mean as line in same subplot
#         axes[i].plot(int_losses_mean, color="red", linewidth=2, label="mean")
#         # show standard deviation as lines around mean
#         axes[i].plot(
#             int_losses_mean + int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         axes[i].plot(
#             int_losses_mean - int_losses_std,
#             color="red",
#             linewidth=1,
#             linestyle="--",
#         )
#         # set x axis label
#         axes[i].set_xlabel("$\\alpha$")
#         # set x axis ticks (0 to 1 in steps of 0.2)
#         axes[i].set_xticks(np.arange(0, 11, 2))
#         # set x axis tick labels (0 to 1 in steps of 0.1)
#         axes[i].set_xticklabels([f"{i / 10:.1f}" for i in range(0, 11, 2)])
#         # set x axis limits
#         axes[i].set_xlim(0, 10)
#         # set y axis label
#         axes[i].set_ylabel("loss")
#         # set y axis limits
#         axes[i].set_ylim(0, 0.25)
#         # set title
#         axes[i].set_title(f"{data} data")
#         # grid
#         axes[i].grid()

#     # set legend
#     axes[i].legend(loc="upper right")

# # set suptitle
# fig.suptitle("Naive interpolation between SWA samples (width=512)")
# # tight layout
# fig.tight_layout()
# # save
# plt.savefig(f"naive_interpolation_losses_{data}_swa_w512.png", dpi=600)
