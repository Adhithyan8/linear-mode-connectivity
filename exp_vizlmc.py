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
# Load data
train_loader = None
test_loader = None
logs = None

# choose 3 random indices from 0 to 50
indices = np.zeros(3, dtype=np.int)
# replace
indices[0] = 47
indices[1] = 46
indices[2] = 17

# reference
ref_idx = 0

# load both models
model0 = MLP(2, width, 1, 1, False)
model1 = MLP(2, width, 1, 1, False)
model2 = MLP(2, width, 1, 1, False)
# load weights
# TODO

# convert models to vectors
vec0 = model2vec(model0)
vec1 = model2vec(model1)
vec2 = model2vec(model2)
origin, x, y, d = get_plane(vec0, vec1, vec2)

xhat = x / torch.norm(x)
yhat = y / torch.norm(y)
xnorm = torch.norm(x).item()
ynorm = torch.norm(y).item()
dnorm = d.item()

x_mid = (dnorm + xnorm) / 3
y_mid = ynorm / 3
x_min, x_max = x_mid - xnorm, x_mid + xnorm
y_min, y_max = y_mid - ynorm, y_mid + ynorm

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


# get the loss grid for the test set
loss_grid_test = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        loss_grid_test[i, j] = get_loss(xx[i, j], yy[i, j], test_loader)

# plot
vmin = -5
vmax = 1
# contour plot on log scale
plt.contourf(
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
cbar = plt.colorbar(
    cm.ScalarMappable(
        norm=colors.LogNorm(vmin=10**vmin, vmax=10**vmax), cmap=cm.coolwarm
    )
)
cbar.ax.set_ylabel("Loss", rotation=270, labelpad=15)
# plot 3 points (0, 0), (1, 0), (d, 1) with names "model0", "model1", "model2"
plt.scatter(
    [0, xnorm, dnorm],
    [0, 0, ynorm],
    c="k",
    s=10,
)
plt.text(
    0,
    0 + 2,
    "$\\theta_A$",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
)
plt.text(
    xnorm,
    0 + 2,
    "$\\theta_B$",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
)
plt.text(
    dnorm,
    ynorm + 2,
    "$\\theta_C$",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
)

plt.title("Loss surface (naive)")
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect("equal")
plt.savefig("lmc_naive.png", dpi=600, bbox_inches="tight")
