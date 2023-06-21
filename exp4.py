import torch
import numpy as np
import matplotlib.pyplot as plt
from architecture.MLP import FCNet
from utils import get_data, plotter, evaluate
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings

warnings.filterwarnings("ignore")

width = 512

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

# choose two random indices from 0 to 49
idx1 = np.random.randint(0, 50)
idx2 = np.random.randint(0, 50)
while idx1 == idx2:
    idx2 = np.random.randint(0, 50)

# replace
idx1 = 33
idx2 = 39

# load both models
model1 = FCNet(2, width, 1, 1)
model2 = FCNet(2, width, 1, 1)

model1.load_state_dict(torch.load(f"models/moons/model_w{width}_{idx1}.pth"))
model2.load_state_dict(torch.load(f"models/moons/model_w{width}_{idx2}.pth"))

average_model = FCNet(2, width, 1, 1)
average_state_dict = OrderedDict()
for key in model1.state_dict():
    average_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2
average_model.load_state_dict(average_state_dict)

# plot all three for all three models
fig, axs = plt.subplots(1, 2)

# set titles for each row
axs[0].set_title(f"model {idx1}")
axs[1].set_title(f"model {idx2}")

# for i, model in enumerate([model1, model2]):
#     # decision boundary
#     model.eval().to(device)
#     x1 = np.linspace(-3, 3, 100)
#     x2 = np.linspace(-3, 3, 100)
#     X1, X2 = np.meshgrid(x1, x2)
#     X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
#     X = torch.from_numpy(X).float().to(device)
#     y = model(X).detach().cpu().numpy().reshape(100, 100)
#     # apply sigmoid
#     y = 1 / (1 + np.exp(-y))

#     w_in = model.layers[0].weight.detach().cpu().numpy()
#     b_in = model.layers[0].bias.detach().cpu().numpy()
#     # plot the lines corresponding to each hidden node
#     for j in range(width):
#         axs[i].plot(
#             x1,
#             -(w_in[j, 0] * x1 + b_in[j]) / w_in[j, 1],
#             linestyle="dotted",
#             alpha=0.2,
#             c=f"C{j}",
#         )
#     axs[i].contourf(x1, x2, y, 3, cmap="RdBu_r", alpha=0.5)
#     # highlight line at 0.5
#     axs[i].contour(x1, x2, y, levels=[0.5], colors="k", linewidths=2)
#     axs[i].set_aspect("equal")
#     axs[i].set_xlim(-3, 3)
#     axs[i].set_ylim(-3, 3)

#     # plot the training data
#     for x_, y_ in train_loader:
#         x_ = x_.numpy()
#         y_ = y_.numpy()
#         axs[i].scatter(x_[y_ == 0, 0], x_[y_ == 0, 1], c="b", s=2)
#         axs[i].scatter(x_[y_ == 1, 0], x_[y_ == 1, 1], c="r", s=2)

# fig.suptitle(f"Decision boundary: width = {width}")
# plt.tight_layout()
# plt.savefig(f"moons_w{width}.png", dpi=600, bbox_inches="tight")

# # plot decision boundary for average model
# fig, axs = plt.subplots(1, 1)

# # decision boundary
# average_model.eval().to(device)
# x1 = np.linspace(-3, 3, 100)
# x2 = np.linspace(-3, 3, 100)
# X1, X2 = np.meshgrid(x1, x2)
# X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
# X = torch.from_numpy(X).float().to(device)
# y = average_model(X).detach().cpu().numpy().reshape(100, 100)
# # apply sigmoid
# y = 1 / (1 + np.exp(-y))

# w_in = average_model.layers[0].weight.detach().cpu().numpy()
# b_in = average_model.layers[0].bias.detach().cpu().numpy()
# # plot the lines corresponding to each hidden node
# for j in range(width):
#     axs.plot(
#         x1,
#         -(w_in[j, 0] * x1 + b_in[j]) / w_in[j, 1],
#         linestyle="dotted",
#         alpha=0.2,
#         c=f"C{j}",
#     )
# axs.contourf(x1, x2, y, 3, cmap="RdBu_r", alpha=0.5)
# # highlight line at 0.5
# axs.contour(x1, x2, y, levels=[0.5], colors="k", linewidths=2)
# axs.set_aspect("equal")
# axs.set_xlim(-3, 3)
# axs.set_ylim(-3, 3)

# # plot the training data
# for x_, y_ in train_loader:
#     x_ = x_.numpy()
#     y_ = y_.numpy()
#     axs.scatter(x_[y_ == 0, 0], x_[y_ == 0, 1], c="b", s=2)
#     axs.scatter(x_[y_ == 1, 0], x_[y_ == 1, 1], c="r", s=2)

# fig.suptitle(f"Average of models {idx1} and {idx2}")
# plt.tight_layout()
# plt.savefig(f"moons_w{width}_average.png", dpi=600, bbox_inches="tight")

# plot the weights
fig, axs = plt.subplots(1, 2)
# size
fig.set_size_inches(8, 4)

# plot the weights
for i, model in enumerate([model1, model2]):
    model.eval().to(device)
    w_in = model.layers[0].weight.detach().cpu().numpy()
    b_in = model.layers[0].bias.detach().cpu().numpy()
    w_out = model.layers[1].weight.detach().cpu().numpy()
    w_in_norm = np.linalg.norm(w_in, axis=1)

    # in second row, plot w_in norm vs w_out
    axs[i].set_xlim(-0.2, 2.5)
    axs[i].set_ylim(-1.25, 1.25)
    # hexbin
    axs[i].hexbin(
        w_in_norm,
        w_out.reshape(-1),
        gridsize=25,
        marginals=True,
        cmap="Blues",
        extent=[-0.2, 2.5, -1.25, 1.25],
        vmin=0,
        vmax=20,    
    )
    # labels
    axs[i].set_xlabel("$\| w_{in} \|_2$")
    axs[i].set_ylabel("$w_{out}$")

    axs[i].set_aspect("auto")
    axs[i].set_title(f"Model {idx1 if i == 0 else idx2}")
    # grid
    axs[i].axhline(0, c="k", alpha=0.2, linestyle="dotted")
    axs[i].axvline(0, c="k", alpha=0.2, linestyle="dotted")
    # colorbar
    cb = fig.colorbar(axs[i].collections[0], ax=axs[i])
    cb.set_label("count")

fig.suptitle(f"width: {width}")
plt.tight_layout()
plt.savefig(f"moons_w{width}_weights_hm.png", dpi=600, bbox_inches="tight")
