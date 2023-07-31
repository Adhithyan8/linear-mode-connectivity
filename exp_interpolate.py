import matplotlib.pyplot as plt
import numpy as np
import torch

from architecture.MLP import MLP, train
from utils import evaluate, get_mnist, get_moons, interpolation_losses, weight_matching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
num_models = 50
depth = 3
epochs = 60

# load data
# train_loader, test_loader = get_moons()
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

    for i in range(1, num_models):
        model = MLP(784, w, depth, 10, layer_norm=True).to(device)
        model.load_state_dict(torch.load(f"models/mnist/model_w{w}_{i}.pth"))
        model.eval()

        model = weight_matching(ref_model, model, depth=depth, layer_norm=True)
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
                int_losses[i, j, :] = interpolation_losses(model_i, model_j, loader)

    np.save(
        f"logs/mnist/perm_int_losses_test_w{width}",
        int_losses,
    )


# visualize naive interpolation losses
for i, width in enumerate(widths):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    int_losses = np.load(f"logs/mnist/perm_int_losses_test_w{width}.npy")

    # compute mean values (average across dim 0 and 1)
    int_losses_mean = int_losses.mean(axis=(0, 1))

    # compute standard deviations (average across dim 0 and 1)
    int_losses_std = int_losses.std(axis=(0, 1))

    # plot individual losses as lines in same subplot
    for j in range(int_losses.shape[0]):
        for k in range(int_losses.shape[1]):
            # plot individual losses as lines in same subplot
            axes.plot(int_losses[j, k], linewidth=0.5, color="grey", alpha=0.1)
    # plot mean as line in same subplot
    axes.plot(int_losses_mean, color="red", linewidth=2, label="mean")
    # show standard deviation as lines around mean
    axes.plot(
        int_losses_mean + int_losses_std,
        color="red",
        linewidth=1,
        linestyle="--",
    )
    axes.plot(
        int_losses_mean - int_losses_std,
        color="red",
        linewidth=1,
        linestyle="--",
    )

    axes.set_xlabel("$\\alpha$")
    axes.set_ylabel("test loss")
    axes.set_xlim(0, 10)
    axes.set_title(f"width {width}")
    axes.grid()
    axes.legend()
    fig.tight_layout()
    plt.savefig(f"perm_interpolation_losses_test_w{width}.png", dpi=600)
    plt.close()


import seaborn as sns

# whitegrid
sns.set_style("whitegrid")

# visualize epsilon
epsilon = np.zeros((int((50 * 49) / 2), len(widths)))
for i, width in enumerate(widths):
    int_losses = np.load(f"logs/mnist/perm_int_losses_test_w{width}.npy")
    idx = 0
    for j in range(int_losses.shape[0]):
        for k in range(int_losses.shape[1]):
            if j < k:
                epsilon[idx, i] = int_losses[j, k, :].max() - max(
                    int_losses[j, k, 0], int_losses[j, k, -1]
                )
                idx += 1

# y axis is 1D vector of epsilon values
epsilon = epsilon.T.flatten()
# x axis is 1D vector of widths
widths = np.repeat(widths, int((11 * 10) / 2))
# log scale
widths = np.log2(widths)
# violin plot
sns.violinplot(
    x=widths,
    y=epsilon,
    cut=0,
    inner="box",
    scale="width",
    color="C2",
)
# set x axis label
plt.xlabel("Hidden layer width")
plt.ylabel("$\\epsilon$")
plt.tight_layout()
plt.savefig(f"perm_epsilon.png", dpi=600)
plt.close()
