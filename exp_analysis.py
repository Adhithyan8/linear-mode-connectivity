import torch
import numpy as np
import matplotlib.pyplot as plt
from architecture.MLP import MLP
from utils import (
    evaluate,
    get_mnist,
    get_moons,
    reduce_model,
)
from matplotlib import colorbar
from matplotlib.colors import Normalize
import seaborn as sns
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim.swa_utils import AveragedModel, SWALR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
train_loader, test_loader = get_moons()

# plot the cosine similarity between incoming weights of node-node pairs
model = MLP(2, 512, 1, 1, False).to(device)
# load weights
# TODO
model.eval().to(device)

# plotting
w_in = model.layers[0].weight.detach().cpu().numpy()
b_in = model.layers[0].bias.detach().cpu().numpy()
w_out = model.layers[1].weight.detach().cpu().numpy()
# add bias as column to w_in
w_in = np.hstack((w_in, b_in.reshape(-1, 1)))
# get cosine similarity between incoming weights of node-node pairs
sim = (
    w_in
    @ w_in.T
    / (
        (
            np.linalg.norm(w_in, axis=1).reshape(-1, 1)
            @ np.linalg.norm(w_in, axis=1).reshape(1, -1)
        )
    )
)

# plot the sim matrix
g = sns.clustermap(
    sim,
    cmap="icefire",
    vmin=-1.0,
    vmax=1.0,
    xticklabels=False,
    yticklabels=False,
    figsize=(16, 16),
    cbar_kws={"label": "cosine similarity"},
    metric="euclidean",
    method="single",
)
# save the figure
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)
g.cax.set_visible(False)
# save the figure
g.savefig(f"moons_w512_sim.png", dpi=600, bbox_inches="tight")

# save the colorbar of icefire cmap
fig, ax = plt.subplots(1, 1, figsize=(0.5, 4))
norm = Normalize(vmin=0, vmax=0.1)
cb1 = colorbar.ColorbarBase(
    ax, cmap="rocket", norm=norm, orientation="vertical", label="$\epsilon$"
)
fig.savefig("rocket_cbar_moons_hyp.png", dpi=600, bbox_inches="tight")


# reduce all models
widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
for width in widths:
    reduced_logs = np.zeros((50, 3))
    for idx in range(50):
        model = MLP(2, width, 1, 1, False).to(device)
        model.load_state_dict(torch.load(f"models/moons/model_w{width}_{idx}.pth"))
        reduced_model, num_nodes = reduce_model(model)
        reduced_model.eval().to(device)
        loss, acc = evaluate(
            reduced_model,
            test_loader,
            criteria=binary_cross_entropy_with_logits,
            output_size=1,
        )
        reduced_logs[idx, 0] = loss
        reduced_logs[idx, 1] = acc
        reduced_logs[idx, 2] = num_nodes
        # save the reduced model
        torch.save(
            reduced_model.state_dict(),
            f"models/moons/reduced_model_w{width}_{idx}.pth",
        )

    np.save(f"logs/moons/reduced_logs_w{width}.npy", reduced_logs)


# config
widths = [8, 16, 64, 512]
num_models = 50
depth = 3
epochs = 60

# data
train_loader, test_loader = get_mnist()

# whitegrid
sns.set_theme(style="whitegrid")

# visualize epsilon after clustering
perm_epsilon_mnist = np.zeros((11, 11, len(widths)))
for i, width in enumerate(widths):
    int_losses = np.load(f"logs/mnist/naive_int_losses_test_swa_w{width}.npy")
    for j in range(int_losses.shape[0]):
        for k in range(int_losses.shape[1]):
            if j > k:
                perm_epsilon_mnist[j, k, i] = perm_epsilon_mnist[k, j, i]
            if j < k:
                perm_epsilon_mnist[j, k, i] = int_losses[j, k, :].max() - max(
                    int_losses[j, k, 0], int_losses[j, k, -1]
                )

for i, width in enumerate(widths):
    g = sns.clustermap(
        perm_epsilon_mnist[:, :, i],
        cmap="rocket",
        vmin=0,
        # vmax=0.1,
        # only show x tick label of 0
        xticklabels=[0] + [""] * 49,
        yticklabels=[0] + [""] * 49,
        row_cluster=False,
        col_cluster=False,
        figsize=(8, 8),
        cbar_kws={"label": "$\epsilon$"},
        metric="euclidean",
        method="single",
    )
    # save the figure
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    # hide the colorbar
    g.cax.set_visible(True)
    # save the figure
    g.savefig(f"swa_sim_mnist_w{width}.png", dpi=600, bbox_inches="tight")


# Lets see if SWA averages are equivalent naive or aligned averaging
widths = [8, 16, 64, 512]
model_losses = np.zeros((len(widths), 11))
swa_model_losses = np.zeros((len(widths), 11))

for i, width in enumerate(widths):
    model = MLP(784, width, depth, 10, True).to(device)
    model.load_state_dict(torch.load(f"models/mnist/model_w{width}_0.pth"))

    # model
    loss, _ = evaluate(model, test_loader, criteria=cross_entropy, output_size=10)
    model_losses[i, 0] = loss

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    swa_model = AveragedModel(model).to(device).train()
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    # as the model is already trained, we start swa right away
    swa_start = 0
    swa_model.update_parameters(model)
    swa_scheduler.step()
    # SWA
    swa_model.eval()
    loss, _ = evaluate(swa_model, test_loader, criteria=cross_entropy, output_size=10)
    swa_model_losses[i, 0] = loss
    swa_model.train()
    for epoch in range(10):
        for x, y in train_loader:
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))
            # Backward pass
            loss.backward()
            optimizer.step()
        # save the model every epoch
        torch.save(model.state_dict(), f"models/mnist/swain_w{width}_{epoch}.pth")
        # evaluate the model
        loss, _ = evaluate(model, test_loader, criteria=cross_entropy, output_size=10)
        model_losses[i, epoch + 1] = loss
        # update the swa model
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        # save the model every epoch
        torch.save(swa_model.state_dict(), f"models/mnist/swa_w{width}_{epoch}.pth")
        # evaluate the swa model
        swa_model.eval()
        loss, _ = evaluate(
            swa_model, test_loader, criteria=cross_entropy, output_size=10
        )
        swa_model_losses[i, epoch + 1] = loss
        swa_model.train()

# save the losses
np.save("logs/mnist/swain_rms_model_losses.npy", model_losses)
np.save("logs/mnist/swa_rms_model_losses.npy", swa_model_losses)
