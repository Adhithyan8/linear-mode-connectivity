import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Choose the dataset
# Options: MNIST, CIFAR10, BLOBS, MOONS, GAUSSIAN, CLASSIFICATION
def get_data(name: str = "BLOBS") -> tuple:
    """
    Get the data loaders for the dataset
    :param name: name of the dataset
    :return: train_loader, test_loader
    """
    if name == "MNIST":
        # MNIST
        # normalize the data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # fetch MNIST dataset and make data loaders
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    elif name == "CIFAR10":
        # CIFAR10
        # normalize the data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # fetch CIFAR10 dataset and make data loaders
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    elif name == "BLOBS":
        # BLOBS
        # generate blobs dataset, split into train and test
        X, y = make_blobs(  # type: ignore
            n_samples=1000,
            centers=2,
            n_features=2,
            random_state=0,
            return_centers=False,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # make data loaders
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
            ),
            batch_size=64,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
            ),
            batch_size=64,
            shuffle=False,
        )

    elif name == "MOONS":
        # MOONS
        # generate moons dataset, split into train and test
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # make data loaders
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
            ),
            batch_size=64,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
            ),
            batch_size=64,
            shuffle=False,
        )

    elif name == "GAUSSIAN":
        # GAUSSIAN
        # generate gaussian dataset, split into train and test
        X, y = make_gaussian_quantiles(
            n_samples=1000, n_features=2, n_classes=2, random_state=0
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # make data loaders
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
            ),
            batch_size=64,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
            ),
            batch_size=64,
            shuffle=False,
        )

    elif name == "CLASSIFICATION":
        # CLASSIFICATION
        # generate classification dataset, split into train and test
        X, y = make_classification(
            n_samples=1000,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # make data loaders
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
            ),
            batch_size=64,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
            ),
            batch_size=64,
            shuffle=False,
        )

    else:
        raise ValueError("Invalid dataset")
    return train_loader, test_loader


def interpolate_models(model1, model2, alpha):
    """
    Interpolate between two models for a given alpha
    """
    # get the state dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # create a new state dict
    new_state_dict = OrderedDict()

    # interpolate the weights and biases
    for k in state_dict1:
        new_state_dict[k] = alpha * state_dict1[k] + (1 - alpha) * state_dict2[k]

    # load the new state dict into a new model
    new_model = copy.deepcopy(model1)
    new_model.load_state_dict(new_state_dict)

    return new_model


# account for scale symmetry (TO DO: account for normalization layers)
def normalize_weights(model):
    """
    Normalize the weights of a model
    """
    # get the state dict
    state_dict = model.state_dict()

    # compute number of layers
    num_layers = len(state_dict) // 2

    # create a new state dict
    new_state_dict = OrderedDict()

    # for each layer except the last
    for i in range(num_layers - 1):
        # get the weight matrix of layer i and i+1
        w_curr = state_dict["layers.{}.weight".format(i)]
        w_next = state_dict["layers.{}.weight".format(i + 1)]

        # add bias of layer i as column to weight of layer i
        b = state_dict["layers.{}.bias".format(i)]
        w_curr = torch.cat((w_curr, b.view(-1, 1)), dim=1)

        # compute norm of rows of layer i
        norm = w_curr.norm(dim=1, keepdim=False)

        # normalize the rows of layer i
        w_curr = w_curr / norm.view(-1, 1)

        # multiply the weight colums of layer i+1 by the norm
        w_next = w_next * norm.view(1, -1)

        # update the old state dict
        state_dict["layers.{}.weight".format(i + 1)] = w_next

        # update the new state dist
        new_state_dict["layers.{}.weight".format(i)] = w_curr[:, :-1]
        new_state_dict["layers.{}.bias".format(i)] = w_curr[:, -1]

    # for the last layer, copy the weights
    new_state_dict["layers.{}.weight".format(num_layers - 1)] = state_dict[
        "layers.{}.weight".format(num_layers - 1)
    ]

    # for the last layer, set the bias to have zero mean
    b = state_dict["layers.{}.bias".format(num_layers - 1)]
    b = b - b.mean()
    new_state_dict["layers.{}.bias".format(num_layers - 1)] = b

    # load the new state dict into a new model
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_state_dict)

    return new_model


def compute_loss(model, loader):
    """
    Compute the loss of a model on a dataset
    """
    # initialize the loss
    loss = 0
    # for each batch
    for X, y in loader:
        # move the data to the device
        X = X.to(device)
        y = y.to(device)

        # compute the loss
        loss += F.cross_entropy(model(X), y).item()

    # return the loss
    return loss / len(loader)


# plot results - get both loss curves and decision boundaries in one plot
# as 3x4 grid, with 1 subplot for loss curves and others for decision boundaries
def plot_results(model1, model2, train_loader, test_loader, name, num_points=11):
    """
    Plot results - get both loss curves and decision boundaries in one plot
    as 3x4 grid, with 1 subplot for loss curves and others for decision boundaries
    """
    # get the state dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # create a new state dict
    new_state_dict = OrderedDict()

    # create a figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    alphas = np.linspace(0, 1, num_points)
    train_losses = []
    test_losses = []

    # get the limits of grid from the data
    X, y = next(iter(test_loader))
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    i, j = 0, 1
    # for each alpha
    for alpha in alphas:
        # interpolate the weights and biases
        for k in state_dict1:
            new_state_dict[k] = alpha * state_dict1[k] + (1 - alpha) * state_dict2[k]

        # load the new state dict into a new model
        new_model = copy.deepcopy(model1)
        new_model.load_state_dict(new_state_dict)

        new_model.eval()

        # compute the loss
        train_loss = compute_loss(new_model, train_loader)
        test_loss = compute_loss(new_model, test_loader)

        # append the loss
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # model is trained with softmax, so apply to output and plot contours of probabilities
        Z = F.softmax(
            new_model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to(device)), dim=1
        )

        # plot the contours in pleasant grey scale
        Z = Z[:, 1].cpu().detach().numpy()
        Z = Z.reshape(xx.shape)
        axes[i, j].contourf(xx, yy, Z, cmap=plt.cm.Greys)

        # plot the data
        for k in range(2):
            axes[i, j].scatter(
                X[y == k, 0].cpu().detach().numpy(),
                X[y == k, 1].cpu().detach().numpy(),
                c="C{}".format(k),
                alpha=0.5,
            )

        # if model has only one hidden layer, plot each neuron in different color
        if len(new_model.layers) == 2:
            # get the weights and biases
            w = new_model.layers[0].weight.cpu().detach().numpy()
            b = new_model.layers[0].bias.cpu().detach().numpy()

            # for each neuron
            for k in range(w.shape[0]):
                # plot the line for the neuron in different color
                axes[i, j].plot(
                    np.linspace(x_min, x_max, 100),
                    -(w[k, 0] * np.linspace(x_min, x_max, 100) + b[k]) / w[k, 1],
                    color="C{}".format(k + 2),
                    linestyle="--",
                )

        # set the title (alpha upto 1 decimal place)
        axes[i, j].set_title(r"$\alpha$ = {:.1f}".format(alpha))

        # set axis limits
        axes[i, j].set_xlim(x_min, x_max)
        axes[i, j].set_ylim(y_min, y_max)

        # increment the indices
        j += 1
        if j == 4:
            i += 1
            j = 0

    # plot the loss, connect the markers, and set the color
    axes[0, 0].plot(alphas, train_losses, color="C0", marker="o", linestyle="-")
    axes[0, 0].plot(alphas, test_losses, color="C1", marker="o", linestyle="--")

    # set the labels
    axes[0, 0].set_xlabel(r"$\alpha$")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")

    # set the legend
    axes[0, 0].legend(["Train", "Test"])

    # set the axis limits
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 4)

    # main title
    fig.suptitle(f"Interpolation: {name}")

    # save the plot
    plt.savefig("plots/{}.png".format(name))

    # close the plot
    plt.close()

    # loss statistics
    stats = loss_stats(train_losses, test_losses)

    # return the stats
    return stats


# loss statistics
def loss_stats(train_losses, test_losses):
    stats = OrderedDict()

    # get the minimum loss
    stats["min_train"] = min(train_losses)
    stats["min_test"] = min(test_losses)

    # get the maximum loss
    stats["max_train"] = max(train_losses)
    stats["max_test"] = max(test_losses)

    # get the average loss
    stats["avg_train"] = sum(train_losses) / len(train_losses)
    stats["avg_test"] = sum(test_losses) / len(test_losses)

    # get eps connectivity (max loss - max of end points)
    stats["eps_train"] = stats["max_train"] - max(train_losses[0], train_losses[-1])
    stats["eps_test"] = stats["max_test"] - max(test_losses[0], test_losses[-1])

    # get loss barrier
    # subtract linear interpolation of end points from losses
    # and get the max of the resulting array
    stats["barrier_train"] = max(
        train_losses - np.linspace(train_losses[0], train_losses[-1], len(train_losses))
    )
    stats["barrier_test"] = max(
        test_losses - np.linspace(test_losses[0], test_losses[-1], len(test_losses))
    )

    return stats


# plot the loss statistics
# given four dictionaries of stats with keys as tuple (i,j)
# save 5 figures of 2x4 grid of plots,
# each containing heatmaps of minimum, maximum loss, avg loss, eps connectivity, and loss barrier
def plot_loss_stats(
    stats_unscaled_naive,
    stats_unscaled_perm,
    stats_rescaled_naive,
    stats_rescaled_perm,
    dataset,
):
    # get the keys
    keys = list(stats_unscaled_naive.keys())

    # get the names of the stats
    names = ["min", "max", "avg", "eps", "barrier"]
    data = ["train", "test"]

    # get the names of the models
    model_names = ["Unscaled Naive", "Unscaled Perm", "Rescaled Naive", "Rescaled Perm"]

    # get the stats
    stats = [
        stats_unscaled_naive,
        stats_unscaled_perm,
        stats_rescaled_naive,
        stats_rescaled_perm,
    ]

    # for each stat
    for k in range(5):
        # create a figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # for each model
        for i in range(4):
            for j in range(2):
                # create a matrix of the stat
                mat = np.zeros((5, 5))
                for m in range(10):
                    mat[keys[m]] = stats[i][keys[m]][f"{names[k]}_{data[j]}"]

                # make lower including diagonal as True
                mask = np.tril(np.ones_like(mat, dtype=np.bool))

                # plot the heatmap with annotations
                sns.heatmap(
                    mat,
                    mask=mask,
                    annot=True,
                    fmt=".2f",
                    vmin=0,
                    vmax=4,
                    cmap="Blues",
                    ax=axes[j, i],
                    cbar=True,
                )

                # set the labels
                axes[j, i].set_xlabel("i")
                axes[j, i].set_ylabel("j")

                # set the title
                axes[j, i].set_title(f"{model_names[i]}: {data[j]}")

                if names[k] == "barrier":
                    # make mat symmetric with diagonal as 0
                    mat = mat + mat.T - np.diag(np.diag(mat))

                    # save the matrix
                    np.save(f"barriers/{dataset}_{model_names[i]}_{data[j]}", mat)

        # main title
        fig.suptitle(f"{names[k]} Loss")

        # save the plot
        plt.savefig(f"plots/{dataset}_{names[k]}.png")

        # close the plot
        plt.close()
