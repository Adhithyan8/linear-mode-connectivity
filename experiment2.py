import numpy as np
import matplotlib.pyplot as plt
import pyvis
import networkx as nx


# create 2*5 subplots with enough space between them
fig, axes = plt.subplots(2, 5, figsize=(20, 8), squeeze=True)

# for widths [3, 4, 5, 6, 12], load the loss barrier and show heatmap, histogram
for i, width in enumerate([3, 4, 5, 6, 12]):
    # load loss barrier
    loss_barrier = np.load(f"logs\sigmoid\gaussian\loss_barriers_s512_w{width}_d1.npy")
    # show heatmap with colorbar
    ax = axes[0, i]
    im = ax.imshow(loss_barrier, cmap="jet", vmin=0, vmax=4)
    ax.set_title("width = {}".format(width))
    # show histogram
    ax = axes[1, i]
    ax.hist(loss_barrier.flatten(), bins=100, range=(0, 8))
    # draw vertical line at 5, 50, 95 percentile
    for p in [5, 50, 95]:
        ax.axvline(np.percentile(loss_barrier, p), color="k", linestyle="--")
    ax.set_title("width = {}".format(width))
    ax.set_xlabel("loss barrier")
# show colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

# save figure
plt.savefig("gaussian_loss_barrier_s512_d1.png", dpi=300)
plt.show()
