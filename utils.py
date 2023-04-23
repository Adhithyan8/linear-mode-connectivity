import copy
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_gaussian_quantiles,
    make_moons,
)
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Choose the dataset
# Options: MNIST, CIFAR10, BLOBS, MOONS, GAUSSIAN, CLASSIFICATION
def get_data(name: str = "blobs", n_samples=512) -> tuple:
    """
    Get the data loaders for the dataset
    :param name: name of the dataset
    :return: train_loader, test_loader
    """
    if name == "mnist":
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

    elif name == "cifar10":
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

    elif name == "blobs":
        # BLOBS
        # generate blobs dataset, split into train and test
        X, y = make_blobs(  # type: ignore
            n_samples=n_samples,
            centers=2,
            n_features=2,
            random_state=0,
            return_centers=False,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

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

    elif name == "moons":
        # MOONS
        # generate moons dataset, split into train and test
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

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

    elif name == "gaussian":
        # GAUSSIAN
        # generate gaussian dataset, split into train and test
        X, y = make_gaussian_quantiles(
            n_samples=n_samples, n_features=2, n_classes=2, random_state=0
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

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

    elif name == "classification":
        # CLASSIFICATION
        # generate classification dataset, split into train and test
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

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
    new_state_dict["layers.{}.bias".format(num_layers - 1)] = b

    # load the new state dict into a new model
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(new_state_dict)

    return new_model


# evaluate the loss and accuracy given a model and loader
def evaluate(model, loader):
    """
    Evaluate the loss and accuracy given a model and loader
    """
    # initialize the loss
    loss = 0
    # initialize the accuracy
    accuracy = 0

    # for each batch
    for X, y in loader:
        y = y.unsqueeze(1).float()

        # move the data to the device
        X = X.to(device)
        y = y.to(device)

        # compute the loss
        loss += F.binary_cross_entropy_with_logits(model(X), y).item()

        # compute the batchwise accuracy (sigmoid of logits)
        accuracy += torch.mean(
            torch.eq(
                torch.round(torch.sigmoid(model(X))),
                y,
            ).float()
        ).item()

    # return the loss and accuracy
    return loss / len(loader), accuracy / len(loader)


# given 2 models and loaders, return a np array of interpolation losses
def interpolation_losses(model1, model2, loader, num_points=11):
    # get the state dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # create a new state dict
    new_state_dict = OrderedDict()

    alphas = np.linspace(0, 1, num_points)
    losses = np.zeros(
        num_points,
    )

    # for each alpha
    for idx in range(len(alphas)):
        # interpolate the weights and biases
        for k in state_dict1:
            new_state_dict[k] = (
                alphas[idx] * state_dict1[k] + (1 - alphas[idx]) * state_dict2[k]
            )

        # load the new state dict into a new model
        new_model = copy.deepcopy(model1)
        new_model.load_state_dict(new_state_dict)

        new_model.eval()

        # compute the loss
        loss, _ = evaluate(new_model, loader)

        # store the loss
        losses[idx] = loss

    return losses


# given a np array of interpolation losses, return the loss barrier
def loss_barrier(losses):
    # compute the loss barrier as max(losses - linspace(losses[0], losses[-1], len(losses)))
    loss_barrier = np.max(losses - np.linspace(losses[0], losses[-1], len(losses)))

    return loss_barrier
