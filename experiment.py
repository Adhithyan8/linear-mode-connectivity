from math import comb

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.scale import LogScale
from scipy.cluster.hierarchy import leaves_list, linkage

from architecture.MLP import FCNet, train
from permute import permute_align
from utils import (
    evaluate,
    get_data,
    interpolation_losses,
    loss_barrier,
    normalize_weights,
)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
datasets = "gaussian"
n_samples = 512
widths = [4, 5, 6, 32, 64, 128, 256, 512]
num_models = 50
depth = 1
epochs = 100

# Load the data
train_loader, test_loader = get_data(name=datasets, n_samples=n_samples)

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
#             lr=0.001,
#             model_name=f"{datasets}/model_s{n_samples}_w{width}_d{depth}_{i}",
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

#     # save the logs
#     np.save(f"logs/sigmoid/{datasets}/logs_s{n_samples}_w{width}_d{depth}", logs)

#     for sym in ["naive", "scale", "perm"]:
#         for data in ["train", "test"]:
#             if sym == "naive":
#                 pass
#             elif sym == "scale" and data == "train":
#                 # pick index of model with lowest train loss as reference
#                 reference_model_idx = np.argmin(logs[:, 0])
#                 print(f"width: {width}, reference model index: {reference_model_idx}")

#                 # normalize weights
#                 for i in range(num_models):
#                     models[i] = normalize_weights(models[i])

#             elif sym == "perm" and data == "train":
#                 models[reference_model_idx].eval().to(device)
#                 # align all other models to this reference model
#                 for i in range(num_models):
#                     if i == reference_model_idx:
#                         continue
#                     models[i].eval().to(device)

#                     models[i] = permute_align(
#                         models[i],
#                         models[reference_model_idx],
#                         train_loader,
#                         epochs=20,
#                         device=device,
#                     )

#             # choose loader
#             if data == "train":
#                 loader = train_loader
#             else:
#                 loader = test_loader

#             # data structure to store interpolation losses
#             int_losses = np.zeros((num_models, num_models, 11))

#             # data structure to store loss barriers
#             barriers = np.zeros((num_models, num_models))

#             # data structure to store max barrier
#             max_barriers = np.zeros((num_models, num_models))

#             # compute interpolation loss for each pair of models
#             # log the results
#             for i in range(num_models):
#                 for j in range(num_models):
#                     if i == j:
#                         continue
#                     if i > j:
#                         int_losses[i, j, :] = int_losses[j, i, :]
#                         barriers[i, j] = barriers[j, i]
#                         max_barriers[i, j] = max_barriers[j, i]
#                         continue
#                     if i < j:
#                         int_losses[i, j, :] = interpolation_losses(
#                             models[i], models[j], loader
#                         )
#                         barriers[i, j] = loss_barrier(int_losses[i, j, :])
#                         max_barriers[i, j] = max(int_losses[i, j, :])

#             np.save(
#                 f"logs/sigmoid/{datasets}/{sym}_int_losses_{data}_s{n_samples}_w{width}_d{depth}",
#                 int_losses,
#             )
#             np.save(
#                 f"logs/sigmoid/{datasets}/{sym}_barriers_{data}_s{n_samples}_w{width}_d{depth}",
#                 barriers,
#             )
#             np.save(
#                 f"logs/sigmoid/{datasets}/{sym}_max_barriers_{data}_s{n_samples}_w{width}_d{depth}",
#                 max_barriers,
#             )

# # visualizing model losses and accuracies
# # create 4*8 subplots with enough space between them, pad the columns
# fig, axes = plt.subplots(4, 8, figsize=(20, 20), squeeze=True, sharey=True)

# # for widths, load the model losses and accuracies and show their histograms
# for i, width in enumerate(widths):
#     model_logs = np.load(
#         f"logs/sigmoid/{datasets}/logs_s{n_samples}_w{width}_d{depth}.npy"
#     )
#     train_losses = model_logs[:, 0]
#     test_losses = model_logs[:, 1]
#     train_accuracies = model_logs[:, 2]
#     test_accuracies = model_logs[:, 3]

#     # show train loss histogram
#     ax = axes[i // 4, i % 4]
#     ax.hist(train_losses, bins=5, color="C0"),
#     ax.set_title("width = {}".format(width))
#     ax.set_xlabel("train loss")
#     # show test loss histogram
#     ax = axes[i // 4, i % 4 + 4]
#     ax.hist(test_losses, bins=5, color="C1")
#     ax.set_title("width = {}".format(width))
#     ax.set_xlabel("test loss")
#     # show train accuracy histogram
#     ax = axes[i // 4 + 2, i % 4]
#     ax.hist(train_accuracies, bins=5, color="C2")
#     ax.set_title("width = {}".format(width))
#     ax.set_xlabel("train accuracy")
#     # show test accuracy histogram
#     ax = axes[i // 4 + 2, i % 4 + 4]
#     ax.hist(test_accuracies, bins=5, color="C3")
#     ax.set_title("width = {}".format(width))
#     ax.set_xlabel("test accuracy")

# # save
# plt.savefig("model_performance.png", dpi=300)

# # close the figure
# plt.close()

# analyze the results
# visualize interpolation losses
for sym in ["naive", "scale", "perm"]:
    for data in ["train", "test"]:
        # create 2*4 subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)

        for i, width in enumerate(widths):
            int_losses = np.load(
                f"logs/sigmoid/{datasets}/{sym}_int_losses_{data}_s{n_samples}_w{width}_d{depth}.npy"
            )

            # compute mean values (average across dim 0 and 1)
            int_losses_mean = int_losses.mean(axis=(0, 1))

            # compute standard deviations (average across dim 0 and 1)
            int_losses_std = int_losses.std(axis=(0, 1))

            # plot individual losses as lines in same subplot
            for j in range(int_losses.shape[0]):
                for k in range(int_losses.shape[1]):
                    axes[i // 4, i % 4].plot(
                        int_losses[j, k], linewidth=0.5, color="grey", alpha=0.1
                    )
            # plot mean as line in same subplot
            axes[i // 4, i % 4].plot(
                int_losses_mean, color="red", linewidth=2, label="mean"
            )
            # show standard deviation as lines around mean
            axes[i // 4, i % 4].plot(
                int_losses_mean + int_losses_std,
                color="red",
                linewidth=1,
                linestyle="--",
            )
            axes[i // 4, i % 4].plot(
                int_losses_mean - int_losses_std,
                color="red",
                linewidth=1,
                linestyle="--",
            )
            # set x axis label
            axes[i // 4, i % 4].set_xlabel("step")
            # set y axis label
            axes[i // 4, i % 4].set_ylabel("loss")
            # set y axis limits
            axes[i // 4, i % 4].set_ylim(0, 2)
            # set title
            axes[i // 4, i % 4].set_title(f"width {width}")
            # grid
            axes[i // 4, i % 4].grid()

        # set legend
        axes[0, 0].legend(loc="upper right")

        # save
        plt.savefig(f"{sym}_interpolation_losses_{data}.png", dpi=300)

        # close the figure
        plt.close()

# visualize barriers
for sym in ["naive", "scale", "perm"]:
    for data in ["train", "test"]:
        # create 2*4 subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        for i, width in enumerate(widths):
            barriers = np.load(
                f"logs/sigmoid/{datasets}/{sym}_barriers_{data}_s{n_samples}_w{width}_d{depth}.npy"
            )

            # condense
            cond_barriers = barriers[np.triu_indices(50, 1)]

            # link
            link = linkage(cond_barriers, method="ward")

            # reorder
            barriers = barriers[leaves_list(link), :]
            barriers = barriers[:, leaves_list(link)]

            # plot heatmap
            im = axes[i // 4, i % 4].imshow(barriers, cmap="viridis", vmin=0, vmax=1.0)
            # title
            axes[i // 4, i % 4].set_title(f"width {width}")
        # common colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

        # save
        plt.savefig(f"{sym}_barriers_{data}.png", dpi=300)

        # close the figure
        plt.close()

# visualize max barriers
for sym in ["naive", "scale", "perm"]:
    for data in ["train", "test"]:
        # create 2*4 subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))

        for i, width in enumerate(widths):
            max_barriers = np.load(
                f"logs/sigmoid/{datasets}/{sym}_max_barriers_{data}_s{n_samples}_w{width}_d{depth}.npy"
            )

            # condense
            cond_max_barriers = max_barriers[np.triu_indices(50, 1)]

            # link
            link = linkage(cond_max_barriers, method="ward")

            # reorder
            max_barriers = max_barriers[leaves_list(link), :]
            max_barriers = max_barriers[:, leaves_list(link)]

            # plot heatmap
            im = axes[i // 4, i % 4].imshow(
                max_barriers,
                cmap="viridis",
                vmin=0,
                vmax=1.0,
            )
            # title
            axes[i // 4, i % 4].set_title(f"width {width}")
        # common colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)

        # save
        plt.savefig(f"{sym}_max_barriers_{data}.png", dpi=300)

        # close the figure
        plt.close()
