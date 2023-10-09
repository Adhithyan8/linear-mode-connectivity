import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import colorbar
from matplotlib.colors import Normalize

from architecture.MLP import MLP, train
from utils import (
    evaluate,
    interpolation_losses,
    weight_matching,
    weight_matching_polar,
    get_moons,
)

import seaborn as sns

sns.set_style("whitegrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
widths = [512]
depth = 1
num_models = 40
epochs = 100

train_loader, test_loader = get_moons()

# training models with different initializations, optimizers
inits = ["normal", "uniform"]
optimizers = ["AdamW", "RMSprop"]
num_mods_each = 10

# consider all combinations of hyperparameters
logs = {}
for init in inits:
    for optim in optimizers:
        for i in range(num_mods_each):
            # initialize the model
            model = MLP(
                input_size=2, width=512, depth=1, output_size=1, layer_norm=False
            ).to(device)
            # reinitialize
            if init == "normal":
                for layer in model.layers:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight)
            elif init == "uniform":
                for layer in model.layers:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_uniform_(layer.weight)
            # define the optimizer
            if optim == "AdamW":
                optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=1e-3)
            elif optim == "RMSprop":
                optimizer = RMSprop(
                    model.parameters(),
                    lr=0.01,
                    alpha=0.99,
                    weight_decay=1e-5,
                    momentum=0.8,
                )

            # train the model
            model.train()
            train(
                model=model,
                loader=train_loader,
                criterion=nn.BCEWithLogitsLoss(),
                optimizer=optimizer,
                lr_scheduler=CosineAnnealingLR(optimizer, T_max=epochs),
                scheduler="epochchwise",
                epochs=epochs,
                model_name=f"moons/model_{init}_{optim}_{i}",
            )

            # evaluate the model
            model.eval()
            train_loss, train_acc = evaluate(
                model,
                train_loader,
                criteria=binary_cross_entropy_with_logits,
                output_size=1,
            )
            test_loss, test_acc = evaluate(
                model,
                test_loader,
                criteria=binary_cross_entropy_with_logits,
                output_size=1,
            )
            logs[f"{init}_{optim}_{i}"] = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }

# save the logs dictionary
torch.save(logs, "logs/moons/logs_hyp.pth")

# load logs
logs = torch.load("logs/moons/logs_hyp.pth")


# get the best model
best_model = None
best = 1.0
for key in logs.keys():
    if logs[key]["test_loss"] < best:
        best = logs[key]["test_loss"]
        best_model = key

print(f"Best model: {best_model}")
print(f"Test loss: {logs[best_model]['test_loss']}")

# initialize the model as ref_model
ref_model = MLP(input_size=2, width=512, depth=1, output_size=1, layer_norm=False).to(
    device
)
# load weights and bias
ref_model.load_state_dict(torch.load(f"models/moons/model_{best_model}.pth"))

int_losses = np.zeros((num_models, num_models, 11))
# align all models to ref_model
for init in inits:
    for optim in optimizers:
        for i in range(num_mods_each):
            # initialize the model
            model = MLP(
                input_size=2,
                width=512,
                depth=1,
                output_size=1,
                layer_norm=False,
            ).to(device)
            # load weights and bias
            model.load_state_dict(
                torch.load(f"models/moons/model_{init}_{optim}_{i}.pth")
            )
            # realign the model
            model = weight_matching_polar(
                ref_model, model, depth=1, layer_norm=False
            )
            # save the model
            torch.save(
                model.state_dict(),
                f"models/moons/perm_polar_model_{init}_{optim}_{i}.pth",
            )

# int_losses = np.zeros((num_models, num_models, 11))
# # align all models to ref_model
# idx_i = 0
# idx_j = 0
# for init in inits:
#     for optim in optimizers:
#         for i in range(num_mods_each):
#             ref_model = MLP(
#                 input_size=2, width=512, depth=1, output_size=1, layer_norm=False
#             ).to(device)
#             # load weights and bias
#             ref_model.load_state_dict(
#                 torch.load(f"models/moons/model_{init}_{optim}_{i}.pth")
#             )
#             # align other models to ref_model
#             for init_ in inits:
#                 for optim_ in optimizers:
#                     for i_ in range(num_mods_each):
#                         # initialize the model
#                         model = MLP(
#                             input_size=2,
#                             width=512,
#                             depth=1,
#                             output_size=1,
#                             layer_norm=False,
#                         ).to(device)
#                         # load weights and bias
#                         model.load_state_dict(
#                             torch.load(f"models/moons/model_{init_}_{optim_}_{i_}.pth")
#                         )
#                         # realign the model
#                         model = weight_matching(
#                             ref_model, model, depth=1, layer_norm=False
#                         )
#                         # # save the model
#                         # torch.save(
#                         #     model.state_dict(),
#                         #     f"models/moons/perm_model_{init}_{optim}_{i}.pth",
#                         # )
#                         # interpolate
#                         if idx_i == idx_j:
#                             int_losses[idx_i, idx_j, :] = np.zeros(11)
#                         if idx_i > idx_j:
#                             int_losses[idx_i, idx_j, :] = int_losses[idx_j, idx_i, :]
#                         if idx_i < idx_j:
#                             int_losses[idx_i, idx_j, :] = interpolation_losses(
#                                 ref_model,
#                                 model,
#                                 test_loader,
#                                 criteria=binary_cross_entropy_with_logits,
#                                 output_size=1,
#                                 num_points=11,
#                             )
#                         idx_j += 1
#             idx_i += 1
#             idx_j = 0

# # save the int_losses
# np.save("logs/moons/perm_int_losses_hyp_all", int_losses)

keys = list(logs.keys())
# interpolate between permuted models
int_losses = np.zeros((num_models, num_models, 11))
for i in range(num_models):
    for j in range(num_models):
        if i == j:
            continue
        if i > j:
            int_losses[i, j, :] = int_losses[j, i, :]
        if i < j:
            # initialize the model
            model_i = MLP(
                input_size=2, width=512, depth=1, output_size=1, layer_norm=False
            ).to(device)
            model_j = MLP(
                input_size=2, width=512, depth=1, output_size=1, layer_norm=False
            ).to(device)
            # load weights and bias
            model_i.load_state_dict(
                torch.load(f"models/moons/perm_polar_model_{keys[i]}.pth")
            )
            model_j.load_state_dict(
                torch.load(f"models/moons/perm_polar_model_{keys[j]}.pth")
            )
            # interpolate
            int_losses[i, j, :] = interpolation_losses(
                model_i,
                model_j,
                test_loader,
                criteria=binary_cross_entropy_with_logits,
                output_size=1,
                num_points=11,
            )
np.save("logs/moons/perm_polar_int_losses_hyp", int_losses)

# load and plot the results
perm_int_losses = np.load("logs/moons/perm_polar_int_losses_hyp.npy")
# mean & std
mean_int_losses = np.mean(perm_int_losses, axis=(0, 1))
std_int_losses = np.std(perm_int_losses, axis=(0, 1))
# q1, q2, q3
q1_int_losses = np.quantile(perm_int_losses, 0.25, axis=(0, 1))
q2_int_losses = np.quantile(perm_int_losses, 0.5, axis=(0, 1))
q3_int_losses = np.quantile(perm_int_losses, 0.75, axis=(0, 1))

# plot
plt.figure(figsize=(4, 4))
for i in range(num_models):
    for j in range(num_models):
        if i < j:
            plt.plot(
                np.arange(0, 1.1, 0.1),
                perm_int_losses[i, j],
                color="grey",
                alpha=0.2,
            )

# q1, q2, q3 as dotted lines
plt.plot(
    np.arange(0, 1.1, 0.1),
    q1_int_losses,
    color="red",
    linestyle="dashed",
    linewidth=1,
)
plt.plot(
    np.arange(0, 1.1, 0.1),
    q2_int_losses,
    color="red",
    linewidth=2,
)
plt.plot(
    np.arange(0, 1.1, 0.1),
    q3_int_losses,
    color="red",
    linestyle="dashed",
    linewidth=1,
)
plt.ylabel("Test loss")
plt.xlabel("$\\alpha$")
plt.xlim(0, 1)
plt.ylim(0, 0.06)
plt.tight_layout()
plt.savefig("perm_polar_int_losses_hyp.png")
plt.close()


# visualize epsilon
perm_epsilon_moons_hyp = np.zeros((num_models, num_models))

int_losses = np.load(f"logs/moons/perm_polar_int_losses_hyp.npy")
for j in range(int_losses.shape[0]):
    for k in range(int_losses.shape[1]):
        if j == k:
            continue
        if j > k:
            perm_epsilon_moons_hyp[j, k] = perm_epsilon_moons_hyp[k, j]
        if j < k:
            perm_epsilon_moons_hyp[j, k] = int_losses[j, k, :].max() - max(
                int_losses[j, k, 0], int_losses[j, k, -1]
            )

# choose 4 colors as row_colors
row_colors = ["C0", "C1", "C2", "C3"]
# repeat each color 10 times
row_colors = np.repeat(row_colors, 10)
# clustermap
g = sns.clustermap(
    perm_epsilon_moons_hyp,
    cmap="rocket",
    vmin=0,
    vmax=0.003,
    xticklabels=[""]*12 + ["Ref"] + [""]*27,
    yticklabels=[""]*12 + ["Ref"] + [""]*27,
    row_cluster=False,
    col_cluster=False,
    row_colors=row_colors,
    col_colors=row_colors,
    figsize=(8, 8),
    method="single"
)
# save the figure
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(False)
# save the figure
g.savefig(f"perm_polar_sim_moons_hyp.png", dpi=600, bbox_inches="tight")


# # save the colorbar of icefire cmap
# fig, ax = plt.subplots(1, 1, figsize=(0.5, 4))
# norm = Normalize(vmin=0, vmax=0.003)
# cb1 = colorbar.ColorbarBase(
#     ax, cmap="rocket", norm=norm, orientation="vertical", label="$\epsilon$"
# )
# fig.savefig("rocket_cbar_moons_hyp.png", dpi=600, bbox_inches="tight")