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

# %%
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
widths = [4, 8, 16, 32, 128, 512]
num_models = 50
depth = 1
epochs = 60

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


# %%
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
            lr=0.001,
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
