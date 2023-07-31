import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from architecture.MLP import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# evaluate the loss and accuracy given a model and loader
def evaluate(model, loader, criteria, output_size=1):
    loss = 0
    accuracy = 0

    for X, y in loader:
        if output_size == 1:
            # unsqueeze the y
            y = y.unsqueeze(1).float()

        X = X.to(device)
        y = y.to(device)
        loss += criteria(model(X), y).item()

        # compute the batchwise accuracy (sigmoid of logits)
        if output_size == 1:
            accuracy += torch.mean(
                torch.eq(
                    torch.round(torch.sigmoid(model(X))),
                    y,
                ).float()
            ).item()
        else:
            accuracy += torch.mean(
                torch.eq(
                    torch.argmax(model(X), dim=1),
                    y,
                ).float()
            ).item()
    # return the loss and accuracy
    return loss / len(loader), accuracy / len(loader)


# given 2 models and loaders, return a np array of interpolation losses
def interpolation_losses(model1, model2, loader, output_size, num_points=11):
    # get the state dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    # create a new state dict
    new_state_dict = OrderedDict()

    alphas = np.linspace(0, 1, num_points)
    losses = np.zeros(
        num_points,
    )
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

        # compute and store the loss
        loss, _ = evaluate(new_model, loader, output_size)
        losses[idx] = loss

    return losses


# given ref model and model, return realigned model
def weight_matching(ref_model, model, depth=3, layer_norm=False):
    width = ref_model.layers[0].weight.shape[0]
    for _ in range(50):
        for l in range(depth):
            # compute cost
            cost = torch.zeros((width, width)).to(device)

            cost += torch.matmul(
                ref_model.layers[int((1 + int(layer_norm)) * l)].weight,
                model.layers[int((1 + int(layer_norm)) * l)].weight.T,
            )
            cost += torch.matmul(
                ref_model.layers[int((1 + int(layer_norm)) * l)].bias.unsqueeze(1),
                model.layers[int((1 + int(layer_norm)) * l)].bias.unsqueeze(0),
            )
            if layer_norm:
                cost += torch.matmul(
                    ref_model.layers[int(2 * l + 1)].weight.unsqueeze(1),
                    model.layers[int(2 * l + 1)].weight.unsqueeze(0),
                )
                cost += torch.matmul(
                    ref_model.layers[int(2 * l + 1)].bias.unsqueeze(1),
                    model.layers[int(2 * l + 1)].bias.unsqueeze(0),
                )
            # next layer
            cost += torch.matmul(
                ref_model.layers[int((1 + int(layer_norm)) * (l + 1))].weight.T,
                model.layers[int((1 + int(layer_norm)) * (l + 1))].weight,
            )

            # get permutation using hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(
                cost.cpu().detach().numpy(), maximize=True
            )
            perm = torch.zeros(cost.shape).to(device)
            perm[row_ind, col_ind] = 1

            # realign model
            model.layers[int((1 + int(layer_norm)) * l)].weight = torch.nn.Parameter(
                torch.matmul(perm, model.layers[int(2 * l)].weight)
            )
            model.layers[int((1 + int(layer_norm)) * l)].bias = torch.nn.Parameter(
                torch.matmul(perm, model.layers[int(2 * l)].bias.unsqueeze(1)).squeeze()
            )
            # change layer norm just like bias
            if layer_norm:
                model.layers[int(2 * l + 1)].weight = torch.nn.Parameter(
                    torch.matmul(
                        perm, model.layers[int(2 * l + 1)].weight.unsqueeze(1)
                    ).squeeze()
                )
                model.layers[int(2 * l + 1)].bias = torch.nn.Parameter(
                    torch.matmul(
                        perm, model.layers[int(2 * l + 1)].bias.unsqueeze(1)
                    ).squeeze()
                )
            # next layer
            model.layers[
                int((1 + int(layer_norm)) * (l + 1))
            ].weight = torch.nn.Parameter(
                torch.matmul(model.layers[int(2 * (l + 1))].weight, perm.T)
            )

    return model


def reduce_model(model, in_threshold=0.0, out_threshold=0.0, sim_threshold=0.99):
    """
    ONLY WORKS FOR 2 LAYER MODELS
    """
    # get the weights
    w_in = model.layers[0].weight.detach().cpu().numpy()
    b_in = model.layers[0].bias.detach().cpu().numpy()
    w_out = model.layers[1].weight.detach().cpu().numpy()
    b_out = model.layers[1].bias.detach().cpu().numpy()

    # remove nodes that share high node-node similarity
    # add bias as column to w_in
    w_in_vec = np.hstack((w_in, b_in.reshape(-1, 1)))
    # get cosine similarity between incoming weights of node-node pairs
    sim = (
        w_in_vec
        @ w_in_vec.T
        / (
            (
                np.linalg.norm(w_in_vec, axis=1).reshape(-1, 1)
                @ np.linalg.norm(w_in_vec, axis=1).reshape(1, -1)
            )
        )
    )
    # consider only upper triangular part and rest as -2
    sim = np.triu(sim, k=1) + np.tril(np.ones_like(sim) * -2, k=0)

    # keep track to nodes considered
    num_nodes_considered = 0
    # loop until all nodes are either kept or removed
    while num_nodes_considered < len(w_in):
        # get the highest similarity pair
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        # within [i, :], [:, i], find j with sim > sim_threshold
        j_0 = np.where(sim[i, :] > sim_threshold)[0]
        j_1 = np.where(sim[:, i] > sim_threshold)[0]
        # union of j_0, j_1
        j_indices = set(j_0).union(set(j_1))
        # keeping i, remove j, k
        w_in = np.delete(w_in, list(j_indices), axis=0)
        b_in = np.delete(b_in, list(j_indices), axis=0)
        # updating w_out
        v1 = w_out[:, i]
        v2 = w_out[:, list(j_indices)]
        n1 = w_in_vec[i, :]
        n2 = w_in_vec[list(j_indices), :]
        lamb = np.linalg.norm(n1) / np.linalg.norm(n2, axis=1)
        w_out[:, i] = v1 + np.sum(lamb.reshape(1, -1) * v2, axis=1)
        w_out = np.delete(w_out, list(j_indices), axis=1)
        # update w_in_vec
        w_in_vec = np.delete(w_in_vec, list(j_indices), axis=0)
        # update sim
        sim = np.delete(sim, list(j_indices), axis=0)
        sim = np.delete(sim, list(j_indices), axis=1)
        # update num_nodes_considered
        num_nodes_considered += len(j_indices) + 1

    # remove low w_out norm nodes
    low_norm_indices = set()
    # get the norm of w_out
    w_out_norm = np.linalg.norm(w_out, axis=0)
    # get the indices of nodes with norm < threshold
    low_norm_indices = low_norm_indices.union(
        set(np.where(w_out_norm < out_threshold)[0])
    )
    # get the norm of w_in_vec
    w_in_norm = np.linalg.norm(w_in_vec, axis=1)
    # get the indices of nodes with norm < threshold (intersection)
    low_norm_indices = low_norm_indices.intersection(
        set(np.where(w_in_norm < in_threshold)[0])
    )

    # remove the nodes
    w_in = np.delete(w_in, list(low_norm_indices), axis=0)
    b_in = np.delete(b_in, list(low_norm_indices), axis=0)
    w_out = np.delete(w_out, list(low_norm_indices), axis=1)
    # update w_in_vec
    w_in_vec = np.delete(w_in_vec, list(low_norm_indices), axis=0)

    # number of remaining nodes
    num_nodes = w_in.shape[0]

    # create new model
    reduced_model = MLP(
        input_size=2, width=num_nodes, depth=1, output_size=1, layer_norm=False
    ).to(device)

    # set weights
    reduced_model.layers[0].weight.data = torch.from_numpy(w_in).float().to(device)
    reduced_model.layers[0].bias.data = torch.from_numpy(b_in).float().to(device)
    reduced_model.layers[1].weight.data = torch.from_numpy(w_out).float().to(device)
    reduced_model.layers[1].bias.data = torch.from_numpy(b_out).float().to(device)

    return reduced_model, num_nodes


def pad_models(model1, model2):
    """
    ONLY WORKS FOR 2 LAYER MODELS
    """
    # pad the smaller model with nodes with zero weights
    width1 = len(model1.layers[0].weight)
    width2 = len(model2.layers[0].weight)
    if width1 > width2:
        # pad model2
        pad = width1 - width2
        model2.layers[0].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model2.layers[0].weight,
                    torch.zeros(pad, model2.layers[0].weight.shape[1]).to(device),
                ],
                dim=0,
            )
        )
        model2.layers[1].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model2.layers[1].weight,
                    torch.zeros(model2.layers[1].weight.shape[0], pad).to(device),
                ],
                dim=1,
            )
        )
        model2.layers[0].bias = torch.nn.Parameter(
            torch.cat([model2.layers[0].bias, torch.zeros(pad).to(device)], dim=0)
        )
    elif width1 < width2:
        # pad model1
        pad = width2 - width1
        model1.layers[0].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model1.layers[0].weight,
                    torch.zeros(pad, model1.layers[0].weight.shape[1]).to(device),
                ],
                dim=0,
            )
        )
        model1.layers[1].weight = torch.nn.Parameter(
            torch.cat(
                [
                    model1.layers[1].weight,
                    torch.zeros(model1.layers[1].weight.shape[0], pad).to(device),
                ],
                dim=1,
            )
        )
        model1.layers[0].bias = torch.nn.Parameter(
            torch.cat([model1.layers[0].bias, torch.zeros(pad).to(device)], dim=0)
        )
    return model1, model2


def get_moons():
    # load data from data/moons.npz
    file = np.load("data/moons.npz")
    X_train = file["X_train"]
    y_train = file["y_train"]
    X_test = file["X_test"]
    y_test = file["y_test"]

    # define train and test loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        ),
        batch_size=256,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        ),
        batch_size=256,
        shuffle=False,
    )
    return train_loader, test_loader


def get_mnist():
    train_loader = DataLoader(
        datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            "data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader
