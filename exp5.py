import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# # Define the loss function
# criterion = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # loss += F.binary_cross_entropy_with_logits(model(X), y).item()
        # loss += F.cross_entropy(model(X), y).item()

from architecture.MLP import (
    MLP,
    FCNet_multiclass,
    FCNet_hyp,
    train,
    train_multiclass,
    train_hyp,
)
from permute import permute_align
from utils import (
    evaluate,
    evaluate_multiclass,
    interpolation_losses,
    loss_barrier,
)
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
widths = [512]
num_models = 40
depth = 1
epochs = 100

# load data from data/moons.npz
file = np.load("data/moons.npz")
X_train = file["X_train"]
y_train = file["y_train"]
X_test = file["X_test"]
y_test = file["y_test"]

# define train and test loaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    ),
    batch_size=256,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    ),
    batch_size=256,
    shuffle=False,
)

# # training models with different initializations, optimizers, WeightDecay and dropout
# inits = [
#     "normal",
#     "uniform",
# ]
# optimizers = ["AdamW", "RMSprop"]
# num_mods_each = 10

# # consider all combinations of hyperparameters
# logs = {}
# for init in inits:
#     for optim in optimizers:
#         for i in range(num_mods_each):
#             # initialize the model
#             model = FCNet(input_size=2, width=512, depth=1, output_size=1).to(device)
#             if init == "normal":
#                 for layer in model.layers:
#                     if isinstance(layer, nn.Linear):
#                         nn.init.kaiming_normal_(layer.weight)
#             elif init == "uniform":
#                 for layer in model.layers:
#                     if isinstance(layer, nn.Linear):
#                         nn.init.kaiming_uniform_(layer.weight)

#             # define the optimizer
#             if optim == "AdamW":
#                 optimizer = torch.optim.AdamW(
#                     model.parameters(), lr=0.1, weight_decay=1e-4
#                 )
#             elif optim == "RMSprop":
#                 optimizer = torch.optim.RMSprop(
#                     model.parameters(), lr=0.1, weight_decay=1e-4
#                 )

#             model.train()
#             # train the model
#             train_hyp(
#                 model,
#                 train_loader,
#                 optimizer,
#                 epochs=epochs,
#                 model_name=f"moons/model_{init}_{optim}_{i}",
#             )

#             # evaluate the model
#             model.eval()
#             train_loss, train_acc = evaluate(model, train_loader)
#             test_loss, test_acc = evaluate(model, test_loader)
#             logs[f"{init}_{optim}_{i}"] = {
#                 "train_loss": train_loss,
#                 "train_acc": train_acc,
#                 "test_loss": test_loss,
#                 "test_acc": test_acc,
#             }

# # save the logs dictionary
# torch.save(logs, "logs/moons/logs_hyp.pth")


# # given ref model and model, return realigned model
# def weight_matching_1layer(ref_model, model):
#     width = ref_model.layers[0].weight.shape[0]
#     for _ in range(50):
#         # compute cost
#         cost = torch.zeros((width, width)).to(device)
#         cost += torch.matmul(ref_model.layers[0].weight, model.layers[0].weight.T)
#         cost += torch.matmul(
#             ref_model.layers[0].bias.unsqueeze(1),
#             model.layers[0].bias.unsqueeze(0),
#         )
#         cost += torch.matmul(
#             ref_model.layers[1].weight.T,
#             model.layers[1].weight,
#         )

#         # get permutation using hungarian algorithm
#         row_ind, col_ind = linear_sum_assignment(
#             cost.cpu().detach().numpy(), maximize=True
#         )
#         perm = torch.zeros(cost.shape).to(device)
#         perm[row_ind, col_ind] = 1

#         # realign model
#         model.layers[0].weight = torch.nn.Parameter(
#             torch.matmul(perm, model.layers[0].weight)
#         )
#         model.layers[0].bias = torch.nn.Parameter(
#             torch.matmul(perm, model.layers[0].bias.unsqueeze(1)).squeeze()
#         )
#         model.layers[1].weight = torch.nn.Parameter(
#             torch.matmul(model.layers[1].weight, perm.T)
#         )

#     return model


# load logs
logs = torch.load("logs/moons/logs_hyp.pth")
# # print
# for key in logs.keys():
#     print(key, logs[key])

# # get the best model
# best_model = None
# best_acc = 0
# for key in logs.keys():
#     if logs[key]["test_acc"] > best_acc:
#         best_acc = logs[key]["test_acc"]
#         best_model = key

# # print the best model
# print(f"Best model: {best_model}")

# # split the best model name
# init, optim, i = best_model.split("_")

# # initialize the model as ref_model
# ref_model = FCNet(input_size=2, width=512, depth=1, output_size=1).to(device)
# # load weights and bias
# ref_model.load_state_dict(torch.load(f"models/moons/model_{best_model}.pth"))

# # align all models to ref_model
# for init in inits:
#     for optim in optimizers:
#         for i in range(num_mods_each):
#             # initialize the model
#             model = FCNet(input_size=2, width=512, depth=1, output_size=1).to(device)
#             # load weights and bias
#             model.load_state_dict(
#                 torch.load(f"models/moons/model_{init}_{optim}_{i}.pth")
#             )

#             # realign the model
#             model = weight_matching_1layer(ref_model, model)

#             # save the model
#             torch.save(
#                 model.state_dict(),
#                 f"models/moons/perm_model_{init}_{optim}_{i}.pth",
#             )

# # # evaluate a model
# # perm_model = FCNet_hyp(input_size=2, width=512, depth=1, output_size=1, dropout=0.0).to(
# #     device
# # )
# # perm_model.load_state_dict(
# #     torch.load("models/moons/perm_model_kaiming_normal_AdamW_0.0005_0.0.pth")
# # )
# # perm_model.eval()
# # train_loss, train_acc = evaluate(perm_model, train_loader)
# # test_loss, test_acc = evaluate(perm_model, test_loader)
# # print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
# # print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

# keys = list(logs.keys())
# # interpolate between permuted models
# int_losses = np.zeros((num_models, num_models, 11))
# for i in range(num_models):
#     for j in range(num_models):
#         if i == j:
#             continue
#         if i > j:
#             int_losses[i, j, :] = int_losses[j, i, :]
#             continue
#         if i < j:
#             # initialize the model
#             model_i = FCNet(input_size=2, width=512, depth=1, output_size=1).to(device)
#             model_j = FCNet(input_size=2, width=512, depth=1, output_size=1).to(device)
#             # load weights and bias
#             model_i.load_state_dict(
#                 torch.load(f"models/moons/perm_model_{keys[i]}.pth")
#             )
#             model_j.load_state_dict(
#                 torch.load(f"models/moons/perm_model_{keys[j]}.pth")
#             )
#             int_losses[i, j, :] = interpolation_losses(model_i, model_j, test_loader)

# np.save("logs/moons/perm_int_losses_hyp", int_losses)

# # load and plot the results
# perm_int_losses = np.load("logs/moons/perm_int_losses_hyp.npy")
# # mean
# mean_int_losses = np.mean(perm_int_losses, axis=(0, 1))
# # std
# std_int_losses = np.std(perm_int_losses, axis=(0, 1))
# # plot
# plt.figure(figsize=(6, 6))
# # plot i, j pairs in grey
# for i in range(num_models):
#     for j in range(num_models):
#         if i == j:
#             continue
#         if i > j:
#             continue
#         if i < j:
#             plt.plot(
#                 np.arange(0, 1.1, 0.1),
#                 perm_int_losses[i, j],
#                 color="grey",
#                 alpha=0.2,
#             )
# plt.plot(
#     np.arange(0, 1.1, 0.1), mean_int_losses, color="red", label="mean", linewidth=3
# )
# # std as dotted lines
# plt.plot(
#     np.arange(0, 1.1, 0.1),
#     mean_int_losses + std_int_losses,
#     color="red",
#     linestyle="dashed",
#     label="std",
#     linewidth=2,
# )
# plt.plot(
#     np.arange(0, 1.1, 0.1),
#     mean_int_losses - std_int_losses,
#     color="red",
#     linestyle="dashed",
#     linewidth=2,
# )
# plt.ylabel("Test loss")
# plt.xlabel("$\\alpha$")
# # x lim
# plt.xlim(0, 1)
# # y lim
# # plt.ylim(0, 0.01)
# # grid
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.savefig("perm_int_losses_hyp.png")
# plt.close()

# visualize epsilon
perm_epsilon_moons_hyp = np.zeros((num_models, num_models))

int_losses = np.load(f"logs/moons/perm_int_losses_hyp.npy")
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
        else:
            continue


import seaborn as sns

# whitegrid
sns.set_style("whitegrid")

# # consider lower triangle
# mask = np.triu(np.ones_like(perm_epsilon_moons_hyp, dtype=bool), k=1)
# # plot violinplot
# plt.figure(figsize=(2, 6))
# sns.violinplot(
#     data=perm_epsilon_moons_hyp[mask],
#     color="C1",
#     inner="box",
#     linewidth=2,
#     scale="width",
#     cut=0,

# )
# # y label
# plt.ylabel("$\epsilon$")
# # x ticks
# plt.xticks([])
# # y ticks
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.ylim(0, 1.0)

# plt.tight_layout()
# plt.savefig("perm_epsilon_moons_hyp.png")
# plt.close()

# keys = list(logs.keys())

# choose 4 colors as row_colors
row_colors = ["C0", "C1", "C2", "C3"]
# repeat each color 10 times
row_colors = np.repeat(row_colors, 10)

g = sns.clustermap(
    perm_epsilon_moons_hyp,
    cmap="rocket",
    vmin=0,
    vmax=0.1,
    xticklabels=False,
    yticklabels=False,
    row_cluster=False,
    col_cluster=False,
    row_colors=row_colors,
    col_colors=row_colors,
    figsize=(8, 8),
)
# save the figure
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
# hide the colorbar
g.cax.set_visible(False)
# save the figure
g.savefig(f"perm_sim_moons_hyp.png", dpi=600, bbox_inches="tight")


# # print keys
# # keys = list(logs.keys())
# # for i, key in enumerate(keys):
# #     print(f"{i}: {key}")

# # show heatmap
# plt.figure(figsize=(16, 16))
# sns.heatmap(
#     perm_epsilon_moons_hyp,
#     cmap="rocket",
#     vmin=0,
#     vmax=0.1,
#     xticklabels=keys,
#     yticklabels=keys,
# )
# # aspect ratio
# plt.gca().set_aspect("equal", adjustable="box")
# plt.savefig("sim_moons_hyp.png")

# # in logs, compute the mean and std of train and test acc. for each setting
# normal_AdamW_train_loss = []
# normal_AdamW_train_acc = []
# normal_AdamW_test_loss = []
# normal_AdamW_test_acc = []
# normal_RMSprop_train_loss = []
# normal_RMSprop_train_acc = []
# normal_RMSprop_test_loss = []
# normal_RMSprop_test_acc = []
# uniform_AdamW_train_loss = []
# uniform_AdamW_train_acc = []
# uniform_AdamW_test_loss = []
# uniform_AdamW_test_acc = []
# uniform_RMSprop_train_loss = []
# uniform_RMSprop_train_acc = []
# uniform_RMSprop_test_loss = []
# uniform_RMSprop_test_acc = []

# for key in logs.keys():
#     if key.startswith("normal_AdamW"):
#         normal_AdamW_train_loss.append(logs[key]["train_loss"])
#         normal_AdamW_train_acc.append(logs[key]["train_acc"])
#         normal_AdamW_test_loss.append(logs[key]["test_loss"])
#         normal_AdamW_test_acc.append(logs[key]["test_acc"])
#     elif key.startswith("normal_RMSprop"):
#         normal_RMSprop_train_loss.append(logs[key]["train_loss"])
#         normal_RMSprop_train_acc.append(logs[key]["train_acc"])
#         normal_RMSprop_test_loss.append(logs[key]["test_loss"])
#         normal_RMSprop_test_acc.append(logs[key]["test_acc"])
#     elif key.startswith("uniform_AdamW"):
#         uniform_AdamW_train_loss.append(logs[key]["train_loss"])
#         uniform_AdamW_train_acc.append(logs[key]["train_acc"])
#         uniform_AdamW_test_loss.append(logs[key]["test_loss"])
#         uniform_AdamW_test_acc.append(logs[key]["test_acc"])
#     elif key.startswith("uniform_RMSprop"):
#         uniform_RMSprop_train_loss.append(logs[key]["train_loss"])
#         uniform_RMSprop_train_acc.append(logs[key]["train_acc"])
#         uniform_RMSprop_test_loss.append(logs[key]["test_loss"])
#         uniform_RMSprop_test_acc.append(logs[key]["test_acc"])
#     else:
#         raise ValueError(f"Unknown key: {key}")

# # print the mean and std
# print("normal_AdamW")
# print(
#     f"Train loss: {np.mean(normal_AdamW_train_loss):.4f} +- {np.std(normal_AdamW_train_loss):.4f}"
# )
# print(
#     f"Train acc: {np.mean(normal_AdamW_train_acc):.4f} +- {np.std(normal_AdamW_train_acc):.4f}"
# )
# print(
#     f"Test loss: {np.mean(normal_AdamW_test_loss):.4f} +- {np.std(normal_AdamW_test_loss):.4f}"
# )
# print(
#     f"Test acc: {np.mean(normal_AdamW_test_acc):.4f} +- {np.std(normal_AdamW_test_acc):.4f}"
# )
# print("normal_RMSprop")
# print(
#     f"Train loss: {np.mean(normal_RMSprop_train_loss):.4f} +- {np.std(normal_RMSprop_train_loss):.4f}"
# )
# print(
#     f"Train acc: {np.mean(normal_RMSprop_train_acc):.4f} +- {np.std(normal_RMSprop_train_acc):.4f}"
# )
# print(
#     f"Test loss: {np.mean(normal_RMSprop_test_loss):.4f} +- {np.std(normal_RMSprop_test_loss):.4f}"
# )
# print(
#     f"Test acc: {np.mean(normal_RMSprop_test_acc):.4f} +- {np.std(normal_RMSprop_test_acc):.4f}"
# )
# print("uniform_AdamW")
# print(
#     f"Train loss: {np.mean(uniform_AdamW_train_loss):.4f} +- {np.std(uniform_AdamW_train_loss):.4f}"
# )
# print(
#     f"Train acc: {np.mean(uniform_AdamW_train_acc):.4f} +- {np.std(uniform_AdamW_train_acc):.4f}"
# )
# print(
#     f"Test loss: {np.mean(uniform_AdamW_test_loss):.4f} +- {np.std(uniform_AdamW_test_loss):.4f}"
# )
# print(
#     f"Test acc: {np.mean(uniform_AdamW_test_acc):.4f} +- {np.std(uniform_AdamW_test_acc):.4f}"
# )
# print("uniform_RMSprop")
# print(
#     f"Train loss: {np.mean(uniform_RMSprop_train_loss):.4f} +- {np.std(uniform_RMSprop_train_loss):.4f}"
# )
# print(
#     f"Train acc: {np.mean(uniform_RMSprop_train_acc):.4f} +- {np.std(uniform_RMSprop_train_acc):.4f}"
# )
# print(
#     f"Test loss: {np.mean(uniform_RMSprop_test_loss):.4f} +- {np.std(uniform_RMSprop_test_loss):.4f}"
# )
# print(
#     f"Test acc: {np.mean(uniform_RMSprop_test_acc):.4f} +- {np.std(uniform_RMSprop_test_acc):.4f}"
# )
