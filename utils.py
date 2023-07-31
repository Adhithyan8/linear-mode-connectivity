import copy
from collections import OrderedDict

import numpy as np
import torch

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
