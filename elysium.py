# experiment 2: make graphs and analyze
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import compute_loss, get_data
from architecture.MLP import FCNet
import torch
from pyvis.network import Network

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# config
samples = [512]
widths = [32]
datasets = ['gaussian']

symmetry = "perm"
data = "train"

for s in samples:
    for w in widths:
        for d in datasets:
            # data loaders
            train_loader, test_loader = get_data(d, s)

            # initialize undirected graph with 50 nodes
            G = nx.Graph()
            G.add_nodes_from(range(50))

            # add feature 'train_loss' to each node
            for node in G.nodes():
                # sample path for model "models\sigmoid\gaussian\model_s128_w4_d1_0.pth"
                path = f"models\sigmoid\{d}\model_s{s}_w{w}_d1_" + str(node) + ".pth"

                # initialize model and load weights
                model = FCNet(2, w, 1, 1).to(device)
                model.load_state_dict(torch.load(path))

                # compute train and test loss
                G.nodes[node]['train_loss'] = compute_loss(model, train_loader)
                G.nodes[node]['test_loss'] = compute_loss(model, test_loader)

                # delete model
                del model
            
            # add edges between nodes
            # sample path "barriers\sigmoid\gaussian\s128_w4_perm_test.npy"
            path = f"barriers\sigmoid\{d}\s{s}_w{w}_{symmetry}_{data}.npy"

            # load loss barriers
            barrier = np.load(path)

# return indices (i,j) in ascending order of barrier values
indices = np.unravel_index(np.argsort(barrier.ravel()), barrier.shape)

# avoid self-loops
indices = [(i,j) for i,j in zip(indices[0], indices[1]) if i != j]

# avoid duplicate edges
edges = []
for i,j in indices:
    if (j,i) not in edges:
        edges.append((i,j))

# create a copy of G
H = G.copy()

# add edges less than threshold
threshold = 0.0001
for i,j in edges:
    if barrier[i,j] < threshold:
        H.add_edge(i,j, weight=barrier[i,j])

# # create a pyvis Network object
# net = Network(height="750px", width="100%")

# # add nodes to the network, color by train loss
# for i in H.nodes():
#     # get the train loss of the node
#     loss = float(H.nodes[i]['train_loss'])
#     # add the node to the network
#     net.add_node(int(i), label=int(i), value=loss, title=f"Train Loss: {loss:.3f}", color="#ff0000", border="white")

# # add edges to the network
# for i,j,data in H.edges(data=True):
#     # get the weight of the edge
#     weight = float(-data['weight'])
#     # add the edge to the network
#     net.add_edge(int(i), int(j), value=weight, title=f"{-weight:.3f}")

# # toggle physics
# net.toggle_physics(True)

# # visualize the network
# net.show("H.html")

# show number of edges
print(f"Number of edges: {len(H.edges())}")

# show number of triangles
print(f"Number of triangles: {nx.triangles(H)}")
