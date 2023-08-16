import torch
import numpy as np
import matplotlib.pyplot as plt
from architecture.MLP import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = MLP(2, 512, 1, 1, False).to(device)
model1.load_state_dict(torch.load(f"models/moons/perm_model_normal_RMSprop_7.pth"))
model2 = MLP(2, 512, 1, 1, False).to(device)
model2.load_state_dict(torch.load(f"models/moons/perm_model_uniform_RMSprop_4.pth"))


def plot_model_units(model, space="weight", name="model"):
    """
    WORKS FOR ONE LAYER ONLY and 2D input
    """
    w_in = model.layers[0].weight.detach().cpu().numpy()
    b_in = model.layers[0].bias.detach().cpu().numpy()
    w_out = model.layers[1].weight.detach().cpu().numpy()
    if space == "weight":
        sizes = b_in.reshape(-1) + 2.0
        colors = w_out.reshape(-1)
        # scatter
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.scatter(
            w_in[:, 0],
            w_in[:, 1],
            c=colors,
            cmap="RdBu_r",
            s=sizes * 15,
            alpha=0.5,
            vmin=-0.5,
            vmax=0.5,
            marker="o",
            linewidth=0,
        )
        # set limits
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        plt.savefig(
            f"{name}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
    elif space == "polar":
        dists = b_in / np.linalg.norm(w_in, axis=1)
        thetas = np.arctan2(w_in[:, 1], w_in[:, 0])
        sizes = np.linalg.norm(w_in, axis=1)
        colors = w_out.reshape(-1)
        # scatter on polar coordinates
        fig, ax = plt.subplots(
            figsize=(16, 16),
            subplot_kw=dict(polar=True),
        )
        ax.scatter(
            thetas,
            dists,
            c=colors,
            cmap="RdBu_r",
            s=sizes * 10,
            alpha=0.5,
            vmin=-0.5,
            vmax=0.5,
            marker="o",
            linewidth=0,
        )
        # set limits
        ax.set_rlim(-2.0, 2.0)
        plt.savefig(
            f"{name}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()


def plot_model_assignment(model1, model2, space="weight", name="model1_model2"):
    w_in1 = model1.layers[0].weight.detach().cpu().numpy()
    b_in1 = model1.layers[0].bias.detach().cpu().numpy()
    w_out1 = model1.layers[1].weight.detach().cpu().numpy()
    w_in2 = model2.layers[0].weight.detach().cpu().numpy()
    b_in2 = model2.layers[0].bias.detach().cpu().numpy()
    w_out2 = model2.layers[1].weight.detach().cpu().numpy()
    if space == "weight":
        sizes1 = b_in1.reshape(-1) + 2.0
        colors1 = w_out1.reshape(-1)
        sizes2 = b_in2.reshape(-1) + 2.0
        colors2 = w_out2.reshape(-1)
        # scatter
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.scatter(
            w_in1[:, 0],
            w_in1[:, 1],
            c=colors1,
            cmap="RdBu_r",
            s=sizes1 * 15,
            alpha=0.5,
            vmin=-0.5,
            vmax=0.5,
            marker="o",
            linewidth=0,
        )
        ax.scatter(
            w_in2[:, 0],
            w_in2[:, 1],
            c=colors2,
            cmap="RdBu_r",
            s=sizes2 * 15,
            alpha=0.5,
            vmin=-0.5,
            vmax=0.5,
            marker="D",
            linewidth=0,
        )
        # draw a quiver connecting the two models
        for i in range(w_in1.shape[0]):
            ax.quiver(
                w_in1[i, 0],
                w_in1[i, 1],
                w_in2[i, 0] - w_in1[i, 0],
                w_in2[i, 1] - w_in1[i, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="grey",
                alpha=0.2,
                width=0.001,
            )
        # set limits
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        plt.savefig(
            f"{name}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
    elif space == "polar":
        dists1 = b_in1 / np.linalg.norm(w_in1, axis=1)
        thetas1 = np.arctan2(w_in1[:, 1], w_in1[:, 0])
        sizes1 = np.linalg.norm(w_in1, axis=1)
        colors1 = w_out1.reshape(-1)
        dists2 = b_in2 / np.linalg.norm(w_in2, axis=1)
        thetas2 = np.arctan2(w_in2[:, 1], w_in2[:, 0])
        sizes2 = np.linalg.norm(w_in2, axis=1)
        colors2 = w_out2.reshape(-1)
        # scatter on polar coordinates
        fig, ax = plt.subplots(
            figsize=(16, 16),
            subplot_kw=dict(polar=True),
        )
        ax.scatter(
            thetas1,
            dists1,
            c=colors1,
            cmap="RdBu_r",
            s=sizes1 * 10,
            alpha=0.5,
            vmin=-0.5,
            vmax=0.5,
            marker="o",
            linewidth=0,
        )
        ax.scatter(
            thetas2,
            dists2,
            c=colors2,
            cmap="RdBu_r",
            s=sizes2 * 10,
            alpha=0.5,
            vmin=-0.5,
            vmax=0.5,
            marker="D",
            linewidth=0,
        )
        # draw a quiver connecting the two models
        for i in range(w_in1.shape[0]):
            ax.quiver(
                thetas1[i],
                dists1[i],
                thetas2[i] - thetas1[i],
                dists2[i] - dists1[i],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="grey",
                alpha=0.2,
                width=0.001,
            )
        # set limits
        ax.set_rlim(-2.0, 2.0)
        plt.savefig(
            f"{name}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()


plot_model_assignment(model1, model2, space="weight", name="w_NR7_UR4_weight")
