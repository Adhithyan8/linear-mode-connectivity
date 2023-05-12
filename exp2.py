import torch
import numpy as np
import matplotlib.pyplot as plt
from architecture.MLP import FCNet
from utils import normalize_weights, get_data, evaluate, plotter
from collections import OrderedDict
from plotly.subplots import make_subplots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# choose settings
datasets = ["moons", "gaussian"]
width = [4, 5, 6, 32, 64, 128, 256, 512]

# choose two random indices from 0 to 49
idx1 = np.random.randint(0, 50)
idx2 = np.random.randint(0, 50)
while idx1 == idx2:
    idx2 = np.random.randint(0, 50)

for dataset in datasets:
    # load the data
    train_loader, test_loader = get_data(dataset, 512)
    for w in width:
        # define and load the models
        model1 = FCNet(2, w, 1, 1)
        model2 = FCNet(2, w, 1, 1)

        model1.load_state_dict(
            torch.load(f"models/sigmoid/{dataset}/model_s512_w{w}_d1_{idx1}.pth")
        )
        model2.load_state_dict(
            torch.load(f"models/sigmoid/{dataset}/model_s512_w{w}_d1_{idx2}.pth")
        )

        # plot the decision boundaries of model1, model2 and the average model
        average_model = FCNet(2, w, 1, 1)
        average_state_dict = OrderedDict()
        for key in model1.state_dict():
            average_state_dict[key] = (
                model1.state_dict()[key] + model2.state_dict()[key]
            ) / 2
        average_model.load_state_dict(average_state_dict)

        plotter(model1, model2, average_model, w, train_loader, title=f"{dataset}_w{w}")
