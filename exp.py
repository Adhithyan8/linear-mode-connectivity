import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colorbar
from matplotlib.colors import Normalize

from architecture.MLP import MLP, train
from utils import (
    get_cifar10,
    evaluate,
    weight_matching,
    weight_matching_test,
    interpolation_losses,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
train_loader, test_loader = get_cifar10()


for optim in ["RMSprop"]:
    for idx in range(10):
        # model
        model = MLP(3 * 32 * 32, 512, 3, 10, True).to(device)
        # criterion
        criterion = torch.nn.CrossEntropyLoss()
        # type of optimizer
        if optim == "AdamW":
            # optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-1
            )
        elif optim == "SGD":
            # optimizer
            optimizer = torch.optim.SGD(
                model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-2
            )
        elif optim == "RMSprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=1e-5,
                weight_decay=0,
                momentum=0,
            )
        elif optim == "Adagrad":
            optimizer = torch.optim.Adagrad(
                model.parameters(), lr=1e-3, weight_decay=1e-2
            )
        # scheduler
        if optim in ["AdamW", "SGD"]:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-2,
                steps_per_epoch=len(train_loader),
                epochs=60,
            )
        elif optim in ["RMSprop"]:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-5,
                steps_per_epoch=len(train_loader),
                epochs=60,
            )
        elif optim == "Adagrad":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-2,
                steps_per_epoch=len(train_loader),
                epochs=60,
                cycle_momentum=False,
            )
        # train
        train(
            model,
            train_loader,
            criterion,
            optimizer,
            lr_scheduler,
            "batchwise",
            epochs=60,
            model_name=f"cifar/model_{optim}_{idx}",
        )

logs = {}
# evaluate
for optim in ["AdamW", "SGD", "RMSprop", "Adagrad"]:
    for idx in range(10):
        # initialize the model
        model = MLP(3 * 32 * 32, 512, 3, 10, True).to(device)
        model.load_state_dict(torch.load(f"models/cifar/model_{optim}_{idx}.pth"))
        model.eval()

        train_loss, train_acc = evaluate(
            model, train_loader, torch.nn.functional.cross_entropy, 10
        )
        test_loss, test_acc = evaluate(
            model, test_loader, torch.nn.functional.cross_entropy, 10
        )
        logs[f"{optim}_{idx}"] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

# save the logs dictionary
torch.save(logs, "logs/cifar/logs_hyp.pth")

# load logs
logs = torch.load("logs/cifar/logs_hyp.pth")

# get the best model
best_model = None
best_loss = 100
for key in logs.keys():
    if logs[key]["test_loss"] < best_loss:
        best_acc = logs[key]["test_loss"]
        best_model = key

print(
    f"Best model: {best_model}, test acc: {best_acc}, test loss: {logs[best_model]['test_loss']}"
)

# initialize the model as ref_model
ref_model = MLP(3 * 32 * 32, 512, 3, 10, True).to(device)
# load weights and bias
ref_model.load_state_dict(torch.load(f"models/cifar/model_{best_model}.pth"))

# align all models to ref_model
for optim in ["AdamW", "SGD", "RMSprop", "Adagrad"]:
    for i in range(10):
        # initialize the model
        model = MLP(3 * 32 * 32, 512, 3, 10, True).to(device)
        # load weights and bias
        model.load_state_dict(torch.load(f"models/cifar/model_{optim}_{i}.pth"))
        # realign the model
        model = weight_matching(ref_model, model, depth=3, layer_norm=True)
        # save the model
        torch.save(
            model.state_dict(),
            f"models/cifar/perm_model_{optim}_{i}.pth",
        )

keys = list(logs.keys())
# interpolate between permuted models
int_losses = np.zeros((40, 40, 11))
for i in range(40):
    for j in range(40):
        if i == j:
            continue
        if i > j:
            int_losses[i, j, :] = int_losses[j, i, :]
        if i < j:
            # initialize the model
            model_i = MLP(3 * 32 * 32, 512, 3, 10, True).to(device)
            model_j = MLP(3 * 32 * 32, 512, 3, 10, True).to(device)
            # load weights and bias
            model_i.load_state_dict(
                torch.load(f"models/cifar/perm_model_{keys[i]}.pth")
            )
            model_j.load_state_dict(
                torch.load(f"models/cifar/perm_model_{keys[j]}.pth")
            )
            # interpolate
            int_losses[i, j, :] = interpolation_losses(
                model_i,
                model_j,
                test_loader,
                criteria=torch.nn.functional.cross_entropy,
                output_size=10,
                num_points=11,
            )
np.save("logs/cifar/perm_int_losses_hyp", int_losses)

import seaborn as sns

sns.set_style("whitegrid")

# load and plot the results
perm_int_losses = np.load("logs/cifar/perm_int_losses_hyp.npy")

# q1, q2, q3
q1_int_losses = np.quantile(perm_int_losses, 0.25, axis=(0, 1))
q2_int_losses = np.quantile(perm_int_losses, 0.5, axis=(0, 1))
q3_int_losses = np.quantile(perm_int_losses, 0.75, axis=(0, 1))

# plot
plt.figure(figsize=(4, 4))
for i in range(40):
    for j in range(40):
        if i < j:
            plt.plot(
                np.arange(0, 1.1, 0.1),
                perm_int_losses[i, j],
                color="grey",
                alpha=0.2,
            )
# q2 as solid line
plt.plot(np.arange(0, 1.1, 0.1), q2_int_losses, color="red", linewidth=2)
# q1, q3 as dotted lines
plt.plot(
    np.arange(0, 1.1, 0.1),
    q1_int_losses,
    color="red",
    linestyle="dashed",
    linewidth=1,
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
plt.ylim(0.0, 3)
plt.tight_layout()
plt.savefig("perm_polar_int_losses_hyp.png")
plt.close()

# visualize epsilon
perm_epsilon_moons_hyp = np.zeros((40, 40))

int_losses = np.load(f"logs/cifar/perm_int_losses_hyp.npy")
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
    vmax=1.0,
    xticklabels=[""] * 24 + ["Ref"] + [""] * 15,
    yticklabels=[""] * 24 + ["Ref"] + [""] * 15,
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
g.savefig(f"perm_sim_cifar_hyp.png", dpi=600, bbox_inches="tight")

# save the colorbar of icefire cmap
fig, ax = plt.subplots(1, 1, figsize=(0.5, 4))
norm = Normalize(vmin=0, vmax=1.0)
cb1 = colorbar.ColorbarBase(
    ax, cmap="rocket", norm=norm, orientation="vertical", label="$\epsilon$",
)
fig.savefig("rocket_cbar_cifar_hyp.png", dpi=600, bbox_inches="tight")


# compute the norm of the parameters
plt.figure(figsize=(6, 6))

# initialize the model
model = MLP(3 * 32 * 32, 512, 3, 10, True).to(device)
norms = np.zeros((10, 4))
optims = ["AdamW", "SGD", "RMSprop", "Adagrad"]
for optim in optims:
    for i in range(10):
        # load weights and bias
        model.load_state_dict(torch.load(f"models/cifar/model_{optim}_{i}.pth"))
        # compute the norm of the parameters
        param_vector = torch.tensor([]).to(device)
        for param in model.parameters():
            param_vector = torch.cat((param_vector, param.flatten()))

        norms[i, optims.index(optim)] = param_vector.norm(p=float("Inf")).item()

# plot stds
plt.bar(
    np.arange(4),
    norms.mean(axis=0),
    yerr=norms.std(axis=0),
    color="grey",
    edgecolor="black",
    linewidth=1,
    alpha=0.5,
)
plt.ylabel("Mean")
plt.xticks(np.arange(4), optims)
plt.title("$L_{\infty}$ norm - parameters")
plt.tight_layout()
plt.savefig("Linf_norms_cifar_hyp.png")
plt.close()
