# %% [markdown]
# Lets start simple: 1 hidden layer networks trained on moons dataset

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage

from architecture.MLP import FCNet, train
from permute import permute_align
from utils import (
    evaluate,
    interpolation_losses,
    loss_barrier,
)
from scipy.optimize import linear_sum_assignment

# %%
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
widths = [4, 8, 16, 32, 128, 512]
num_models = 50
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

for width in widths:
    # data structure to store losses and accuracies
    logs = np.zeros((num_models, 4))

    # Define and train many models
    models = []
    for i in range(num_models):
        model = FCNet(input_size=2, width=width, depth=depth, output_size=1)
        train(
            model,
            train_loader,
            epochs=epochs,
            lr=0.1,
            model_name=f"moons/model_w{width}_{i}",
        )
        models.append(model)

        # evaluate
        model.eval()

        train_loss, train_acc = evaluate(model, train_loader)
        test_loss, test_acc = evaluate(model, test_loader)

        logs[i, 0] = train_loss
        logs[i, 1] = test_loss
        logs[i, 2] = train_acc
        logs[i, 3] = test_acc

        # save the logs
        np.save(f"logs/moons/logs_w{width}", logs)

# visualizing model losses and accuracies
fig, axes = plt.subplots(2, 2, figsize=(5, 7), sharey=True)
# title
fig.suptitle("2-layer MLPs on moons")

# for widths, load the model losses and accuracies and show their stacked histograms
train_losses = np.zeros((num_models, len(widths)))
test_losses = np.zeros((num_models, len(widths)))
train_accs = np.zeros((num_models, len(widths)))
test_accs = np.zeros((num_models, len(widths)))
                      
for i, width in enumerate(widths):
    model_logs = np.load(
        f"logs/moons/logs_w{width}.npy"
    )
    train_losses[:, i] = model_logs[:, 0]
    test_losses[:, i] = model_logs[:, 1]
    train_accs[:, i] = model_logs[:, 2]
    test_accs[:, i] = model_logs[:, 3]

# show train loss
axes[0, 0].hist(train_losses, bins=10, stacked=True, label=widths, range=(0, 0.3), alpha=0.8)
axes[0, 0].set_title("Train loss")

axes[0, 1].hist(test_losses, bins=10, stacked=True, label=widths, range=(0, 0.3), alpha=0.8)
axes[0, 1].set_title("Test loss")
axes[0, 1].legend(title="widths")
axes[1, 0].hist(train_accs, bins=10, stacked=True, label=widths, range=(0.85, 1), alpha=0.8)
axes[1, 0].set_title("Train accuracy")

axes[1, 1].hist(test_accs, bins=10, stacked=True, label=widths, range=(0.85, 1), alpha=0.8)
axes[1, 1].set_title("Test accuracy")


# padding
fig.tight_layout(pad=1.0)
# save
plt.savefig("model_performance.png", dpi=300)


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

for w in widths:
    ref_model = FCNet(2, w, 1, 1).to(device)
    ref_model.load_state_dict(torch.load(f"models/moons/model_w{w}_0.pth"))
    ref_model.eval()

    for i in range(1, num_models):
        model = FCNet(2, w, 1, 1).to(device)
        model.load_state_dict(torch.load(f"models/moons/model_w{w}_{i}.pth"))
        model.eval()

        model = weight_matching(ref_model, model)
        torch.save(model.state_dict(), f"models/moons/perm_model_w{w}_{i}.pth")

for width in widths:
    models = []
    for i in range(num_models):
        model = FCNet(2, width, 1, 1).to(device)
        model.load_state_dict(torch.load(f"models/moons/model_w{width}_{i}.pth"))
        models.append(model)

    for data in ["train", "test"]:
        # choose loader
        if data == "train":
            loader = train_loader
        else:
            loader = test_loader

        # data structure to store interpolation losses
        int_losses = np.zeros((num_models, num_models, 11))

        # data structure to store loss barriers
        barriers = np.zeros((num_models, num_models))

        # data structure to store max barrier
        max_barriers = np.zeros((num_models, num_models))

        # compute interpolation loss for each pair of models
        # log the results
        for i in range(num_models):
            for j in range(num_models):
                if i == j:
                    continue
                if i > j:
                    int_losses[i, j, :] = int_losses[j, i, :]
                    barriers[i, j] = barriers[j, i]
                    max_barriers[i, j] = max_barriers[j, i]
                    continue
                if i < j:
                    int_losses[i, j, :] = interpolation_losses(
                        models[i], models[j], loader
                    )
                    barriers[i, j] = loss_barrier(int_losses[i, j, :])
                    max_barriers[i, j] = max(int_losses[i, j, :])

        np.save(
            f"logs/moons/naive_int_losses_{data}_w{width}",
            int_losses,
        )
        np.save(
            f"logs/moons/naive_barriers_{data}_w{width}",
            barriers,
        )
        np.save(
            f"logs/moons/naive_max_barriers_{data}_w{width}",
            max_barriers,
        )

# visualize naive interpolation losses
for data in ["train", "test"]:
    # create 3*2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(7, 8), sharex=True, sharey=True)

    for i, width in enumerate(widths):
        int_losses = np.load(
            f"logs/moons/naive_int_losses_{data}_w{width}.npy"
        )

        # compute mean values (average across dim 0 and 1)
        int_losses_mean = int_losses.mean(axis=(0, 1))

        # compute standard deviations (average across dim 0 and 1)
        int_losses_std = int_losses.std(axis=(0, 1))

        # plot individual losses as lines in same subplot
        for j in range(int_losses.shape[0]):
            for k in range(int_losses.shape[1]):
                axes[i // 2, i % 2].plot(
                    int_losses[j, k], linewidth=0.5, color="grey", alpha=0.1
                )
        # plot mean as line in same subplot
        axes[i // 2, i % 2].plot(
            int_losses_mean, color="red", linewidth=2, label="mean"
        )
        # show standard deviation as lines around mean
        axes[i // 2, i % 2].plot(
            int_losses_mean + int_losses_std,
            color="red",
            linewidth=1,
            linestyle="--",
        )
        axes[i // 2, i % 2].plot(
            int_losses_mean - int_losses_std,
            color="red",
            linewidth=1,
            linestyle="--",
        )
        # set x axis label
        axes[i // 2, i % 2].set_xlabel("$\\alpha$")
        # set x axis ticks (0 to 1 in steps of 0.2)
        axes[i // 2, i % 2].set_xticks(np.arange(0, 11, 2))
        # set x axis tick labels (0 to 1 in steps of 0.1)
        axes[i // 2, i % 2].set_xticklabels(
            [f"{i / 10:.1f}" for i in range(0, 11, 2)]
        )
        # set x axis limits
        axes[i // 2, i % 2].set_xlim(0, 10)
        # set y axis label
        axes[i // 2, i % 2].set_ylabel("loss")
        # set y axis limits
        axes[i // 2, i % 2].set_ylim(0, 2)
        # set title
        axes[i // 2, i % 2].set_title(f"width {width}")
        # grid
        axes[i // 2, i % 2].grid()

    # set legend
    axes[0, 0].legend(loc="upper right")

    # set suptitle
    fig.suptitle(f"Interpolation between weights: {data} loss")
    # tight layout
    fig.tight_layout()
    # save
    plt.savefig(f"naive_interpolation_losses_{data}.png", dpi=600)

# visualize naive interpolation losses
epsilon = np.zeros((2,6))
epsilon_std = np.zeros((2,6))
for data in ["test"]:
    for i, width in enumerate(widths):
        int_losses = np.load(
            f"logs/moons/naive_int_losses_{data}_w{width}.npy"
        )
        # compute mean values (average across dim 0 and 1)
        int_losses_mean = int_losses.mean(axis=(0, 1))
        # compute standard deviations (average across dim 0 and 1)
        int_losses_std = int_losses.std(axis=(0, 1))
        # compute epsilon
        if data == "train":
            epsilon[0,i] = int_losses_mean[5]
            epsilon_std[0,i] = int_losses_std[5]
        else:
            epsilon[1,i] = int_losses_mean[5]
            epsilon_std[1,i] = int_losses_std[5]

# visualize
fig, ax = plt.subplots(1,1,figsize=(7,4))
# x axis in log scale
ax.set_xscale("log")
# plot epsilon
ax.plot(widths, epsilon[1], label="test", color="blue", marker="o")
# plot standard deviation as error bars
ax.errorbar(
    widths,
    epsilon[1],
    yerr=epsilon_std[1],
    color="blue",
    linewidth=1,
    linestyle="none",
    capsize=2,
)
# set x axis label
ax.set_xlabel("width")
# set x axis ticks
ax.set_xticks(widths)
# set x axis tick labels
ax.set_xticklabels(widths)
# set x axis limits
ax.set_xlim(3.5, 580)
# set y axis label
ax.set_ylabel("$\\epsilon$")
# grid
ax.grid()
# set suptitle
fig.suptitle("$\\epsilon$-linear mode connectivity")
# tight layout
# fig.tight_layout()
# save
plt.savefig(f"naive_epsilon.png", dpi=600)
