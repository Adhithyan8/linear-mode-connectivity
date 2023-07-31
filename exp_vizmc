from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors
from torch.nn.functional import binary_cross_entropy_with_logits

from architecture.MLP import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# convert model parameters to vectors
def model2vec(model):
    vec = []
    for p in model.parameters():
        vec.append(p.view(-1))
    vec = torch.cat(vec).view(-1, 1)
    return vec


# convert vectors to model parameters
def vec2model(vec, model):
    vec = vec.view(-1)
    vec = vec.to(device)
    model_dict = model.state_dict()
    state_dict = OrderedDict()
    start = 0
    for k, v in model_dict.items():
        end = start + v.numel()
        state_dict[k] = vec[start:end].view(v.shape)
        start = end
    model.load_state_dict(state_dict)
    return model


# given three vectors, return the x-y plane's basis they define and all 3 points in new basis
def get_plane(v0, v1, v2):
    v0 = v0.view(-1, 1)
    v1 = v1.view(-1, 1)
    v2 = v2.view(-1, 1)

    origin = v0
    x = v1 - origin  # v1 = origin + x
    # d = (v2 - origin).x / |x|
    xnorm = torch.norm(x)
    d = torch.matmul(x.T, v2 - origin) / xnorm
    # v2 = origin + y + d.x/|x|
    y = v2 - origin - d * x / xnorm

    return origin, x, y, d


# config
width = None
train_loader = None
test_loader = None
logs = None

# choose 2 indices
indices = np.zeros(2, dtype=np.int)
# replace
indices[0] = 10
indices[1] = 23

# load both models
model0 = MLP(2, width, 1, 1, False)
model1 = MLP(2, width, 1, 1, False)

# load weights
# TODO

# model2 is average of model0 and model1 + noise
model2 = MLP(2, width, 1, 1, False)
model2_dict = model2.state_dict()
for k, v in model2_dict.items():
    noise = torch.randn(v.shape) * 0.01
    model2_dict[k] = (model0.state_dict()[k] + model1.state_dict()[k]) / 2 + noise
model2.load_state_dict(model2_dict)

# convert models to vectors
vec0 = model2vec(model0)
vec1 = model2vec(model1)
vec2 = model2vec(model2)
origin, x, y, d = get_plane(vec0, vec1, vec2)

xhat = x / torch.norm(x)
yhat = y / torch.norm(y)
xnorm_ = torch.norm(x).item()
ynorm_ = torch.norm(y).item()
dnorm_ = d.item()

x_mid = (dnorm_ + xnorm_) / 3
y_mid = ynorm_ / 3
x_min, x_max = x_mid - xnorm_, x_mid + xnorm_
y_min, y_max = y_mid - 10, y_mid + 10

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)

# model at grid[i, j] is model2vec(vec0 + i * x + j * y)
model_ = MLP(2, width, 1, 1, False)


def get_model(i, j):
    return vec2model(vec0 + i * xhat + j * yhat, model_)


# compute loss at grid[i, j]
def get_loss(i, j, loader):
    model = get_model(i, j)
    model.to(device).eval()
    loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1).float()
        loss += binary_cross_entropy_with_logits(model(x), y, reduction="mean")
        loss /= len(loader)
    return loss.item()


# compute loss at grid[i, j] for all i, j
loss_grid_train = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        loss_grid_train[i, j] = get_loss(xx[i, j], yy[i, j], train_loader)

loss_grid_test = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        loss_grid_test[i, j] = get_loss(xx[i, j], yy[i, j], test_loader)

# plot train and test loss landscapes
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
vmin = -5
vmax = 1

# contour plot on log scale
ax[0].contourf(
    xx,
    yy,
    np.log(loss_grid_train),
    cmap=cm.coolwarm,
    levels=40,
    interpolation="none",
    vmin=vmin,
    vmax=vmax,
)

# contour plot on log scale
ax[1].contourf(
    xx,
    yy,
    np.log(loss_grid_test),
    cmap=cm.coolwarm,
    levels=40,
    interpolation="none",
    vmin=vmin,
    vmax=vmax,
)

# colorbar in log scale
cbar = fig.colorbar(
    cm.ScalarMappable(
        norm=colors.LogNorm(vmin=10**vmin, vmax=10**vmax), cmap=cm.coolwarm
    ),
    ax=ax[0],
)
cbar.ax.set_ylabel("Train Loss", rotation=270, labelpad=15)
cbar = fig.colorbar(
    cm.ScalarMappable(
        norm=colors.LogNorm(vmin=10**vmin, vmax=10**vmax), cmap=cm.coolwarm
    ),
    ax=ax[1],
)
cbar.ax.set_ylabel("Test Loss", rotation=270, labelpad=15)

ax[0].scatter(
    [0, xnorm_, dnorm_],
    [0, 0, ynorm_],
    c=["red", "green", "blue"],
    s=10,
)
ax[1].scatter(
    [0, xnorm_, dnorm_],
    [0, 0, ynorm_],
    c=["red", "green", "blue"],
    s=10,
)
ax[0].set_title("Train Loss Landscape")
ax[1].set_title("Test Loss Landscape")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[0].set_aspect("equal")
ax[1].set_aspect("equal")
plt.show()


def sample_model(vec0, vec1, model2):
    vec2 = model2vec(model2).to(device)
    vec0 = vec0.to(device)
    vec1 = vec1.to(device)

    norm_2_0 = torch.norm(vec2 - vec0).item()
    norm_2_1 = torch.norm(vec2 - vec1).item()
    # pick a number between 0 and 1
    alpha = torch.rand(1).item()
    # transition value
    trans = norm_2_0 / (norm_2_0 + norm_2_1)
    # pick a model based on alpha
    if alpha <= trans:
        # sample a model along the path 0 -> 2
        alpha = alpha / trans
        model2 = vec2model((1 - alpha) * vec0 + alpha * vec2, model2)
    else:
        # sample a model along the path 2 -> 1
        alpha = (alpha - trans) / (1 - trans)
        model2 = vec2model((1 - alpha) * vec2 + alpha * vec1, model2)
    return model2


# re-train model2 such that expected loss along model 0 -> 1 -> 2 is minimized
for _ in range(100):
    # Define the loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Define the optimizer
    optimizer = torch.optim.AdamW(model2.parameters(), lr=1e-2)
    # sample a model along the path 0 -> 1 -> 2
    model2 = sample_model(vec0, vec1, model2)
    # Train the model
    model2.to(device)
    model2.train()
    for epoch in range(1):
        for x, y in train_loader:
            # model has 1 output
            y = y.unsqueeze(1).float()
            # Forward pass
            optimizer.zero_grad()
            y_pred = model2(x.to(device))
            loss = criterion(y_pred, y.to(device))
            # Backward pass
            loss.backward()
            optimizer.step()

# Repeat from converting to vec. to plotting
# TODO
