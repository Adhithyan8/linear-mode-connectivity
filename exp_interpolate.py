import matplotlib.pyplot as plt
import numpy as np
import torch

from architecture.MLP import MLP, train
from utils import (
    evaluate,
    get_mnist,
    get_moons,
    interpolation_losses,
    weight_matching,
    weight_matching_test,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
num_models = 50
depth = 3
epochs = 60

# load data
train_loader, test_loader = get_mnist()

# train and evaluate models
for width in widths:
    # data structure to store losses and accuracies
    logs = np.zeros((num_models, 4))

    # define and train many models
    for i in range(num_models):
        model = MLP(
            input_size=784, width=width, depth=depth, output_size=10, layer_norm=True
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=epochs
        )
        train(
            model=model,
            loader=train_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scheduler="batchwise",
            epochs=epochs,
            model_name=f"mnist/model_w{width}_{i}",
        )

        # evaluate
        model.eval()
        criteria = torch.nn.functional.cross_entropy
        train_loss, train_acc = evaluate(model, train_loader, criteria, output_size=10)
        test_loss, test_acc = evaluate(model, test_loader, criteria, output_size=10)

        logs[i, 0] = train_loss
        logs[i, 1] = test_loss
        logs[i, 2] = train_acc
        logs[i, 3] = test_acc
    # save the logs
    np.save(f"logs/mnist/logs_w{width}", logs)


for w in widths:
    ref_model = MLP(784, w, depth, 10, layer_norm=True).to(device)
    ref_model.load_state_dict(torch.load(f"models/mnist/model_w{w}_0.pth"))
    ref_model.eval()

    for i in range(num_models):
        model = MLP(784, w, depth, 10, layer_norm=True).to(device)
        model.load_state_dict(torch.load(f"models/mnist/model_w{w}_{i}.pth"))
        model.eval()

        model, swaps = weight_matching_test(
            ref_model, model, depth=depth, layer_norm=True
        )
        torch.save(model.state_dict(), f"models/mnist/perm_model_w{w}_{i}.pth")


for width in widths:
    loader = test_loader

    # data structure to store interpolation losses
    int_losses = np.zeros((num_models, num_models, 11))

    # compute interpolation loss for each pair of models
    for i in range(num_models):
        for j in range(num_models):
            if i > j:
                int_losses[i, j, :] = int_losses[j, i, :]
            if i < j:
                model_i = MLP(
                    input_size=784,
                    width=width,
                    depth=3,
                    output_size=10,
                    layer_norm=True,
                ).to(device)
                model_i.load_state_dict(
                    torch.load(f"models/mnist/perm_model_w{width}_{i}.pth")
                )
                model_j = MLP(
                    input_size=784,
                    width=width,
                    depth=3,
                    output_size=10,
                    layer_norm=True,
                ).to(device)
                model_j.load_state_dict(
                    torch.load(f"models/mnist/perm_model_w{width}_{j}.pth")
                )
                int_losses[i, j, :] = interpolation_losses(
                    model_i,
                    model_j,
                    loader,
                    criteria=torch.nn.functional.cross_entropy,
                    output_size=10,
                )

    np.save(
        f"logs/mnist/perm_int_losses_test_w{width}",
        int_losses,
    )

import seaborn as sns

# whitegrid
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()
# visualize naive interpolation losses
for i, width in enumerate(widths):
    int_losses = np.load(f"logs/moons/naive_int_losses_test_w{width}.npy")

    # compute the interquartiles
    q1 = np.quantile(int_losses, 0.25, axis=(0, 1))
    q2 = np.quantile(int_losses, 0.5, axis=(0, 1))
    q3 = np.quantile(int_losses, 0.75, axis=(0, 1))

    # plot individual losses as lines in same subplot
    for j in range(int_losses.shape[0]):
        for k in range(int_losses.shape[1]):
            # plot individual losses as lines in same subplot
            axes[i].plot(int_losses[j, k], linewidth=0.5, color="grey", alpha=0.1)

    # plot interquartiles as lines in same subplot
    axes[i].plot(q1, color="red", linewidth=1, linestyle="--")
    axes[i].plot(q2, color="red", linewidth=2, label="median")
    axes[i].plot(q3, color="red", linewidth=1, linestyle="--")

    axes[i].set_xlabel("$\\alpha$")
    axes[i].set_xlim(0, 10)
    axes[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes[i].set_ylabel("Test loss")
    axes[i].set_title(f"Hidden layer width {width}")
fig.tight_layout()
plt.savefig(f"moons_naive_interpolation_losses_test.png", dpi=600)
plt.close()

# naive epsilon
epsilon_naive_moons = np.zeros((int((50 * 49) / 2), len(widths)))
for i, width in enumerate(widths):
    int_losses = np.load(f"logs/moons/naive_int_losses_test_w{width}.npy")
    idx = 0
    for j in range(int_losses.shape[0]):
        for k in range(int_losses.shape[1]):
            if j < k:
                epsilon_naive_moons[idx, i] = int_losses[j, k, :].max() - max(
                    int_losses[j, k, 0], int_losses[j, k, -1]
                )
                idx += 1

# perm epsilon
epsilon_perm_moons = np.zeros((int((50 * 49) / 2), len(widths)))
for i, width in enumerate(widths):
    int_losses = np.load(f"logs/moons/reduced_int_losses_test_w{width}.npy")
    idx = 0
    for j in range(int_losses.shape[0]):
        for k in range(int_losses.shape[1]):
            if j < k:
                epsilon_perm_moons[idx, i] = int_losses[j, k, :].max() - max(
                    int_losses[j, k, 0], int_losses[j, k, -1]
                )
                idx += 1

# plot moons
eps = np.concatenate((epsilon_naive_moons.T.flatten(), epsilon_perm_moons.T.flatten()))
ws = np.concatenate(
    (
        np.repeat(widths, int((50 * 49) / 2)),
        np.repeat(widths, int((50 * 49) / 2)),
    )
)
ws = np.log2(ws)
hue = np.concatenate(
    (
        np.repeat("Full", int((50 * 49) * 9 / 2)),
        np.repeat("Reduced", int((50 * 49) * 9 / 2)),
    )
)

# boxplot
fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(
    x=ws,
    y=eps,
    hue=hue,
    split=True,
    inner="quartile",
    cut=0,
    scale="width",
    palette=["C0", "C3"],
)
plt.legend()
plt.xticks(np.arange(0, 9, 1), labels=widths)
plt.xlabel("Hidden layer width")
plt.ylabel("$\\epsilon$")
plt.tight_layout()
plt.savefig(f"compare_epsilon_reduced.png", dpi=600)
plt.close()
