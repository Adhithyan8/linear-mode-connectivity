import numpy as np
import torch

from architecture.MLP import FCNet, train
from permute import permute_align
from utils import get_data, normalize_weights, plot_loss_stats, plot_results

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = "embed"  # "permute" or "embed"

if experiment == "permute":
    # datasets
    datasets = ["BLOBS", "MOONS", "GAUSSIAN", "CLASSIFICATION"]

    for dataset in datasets:
        # Load the data
        train_loader, test_loader = get_data(name=dataset)

        # config
        num_models = 20
        width = 6
        depth = 1
        epochs = 100

        # Define and train many models
        models = []
        for i in range(num_models):
            model = FCNet(input_size=2, width=width, depth=depth, output_size=2)
            train(
                model,
                train_loader,
                epochs=epochs,
                lr=0.001,
                model_name=f"model_w{width}_d{depth}_{dataset}_{i}",
            )
            models.append(model)

        # or load the models from disk
        models = []
        for i in range(num_models):
            model = FCNet(input_size=2, width=width, depth=depth, output_size=2)
            model.load_state_dict(
                torch.load(f"models/model_w{width}_d{depth}_{dataset}_{i}.pth")
            )
            models.append(model)

        barriers_unscaled_naive = dict()
        barriers_unscaled_perm = dict()
        barriers_rescaled_naive = dict()
        barriers_rescaled_perm = dict()

        for i in range(num_models):
            for j in range(i + 1, num_models):
                models[i].eval().to(device)
                models[j].eval().to(device)

                # store the loss barriers
                barriers_unscaled_naive[(i, j)] = plot_results(
                    models[i],
                    models[j],
                    train_loader,
                    test_loader,
                    name=f"{dataset}_model{i}_model{j}_unscaled_naive",
                    save=False,
                )

                # align the models
                pi_model = permute_align(
                    models[i], models[j], train_loader, epochs=20, device=device
                )
                torch.save(
                    pi_model,
                    f"models/pi_model{i}_model{j}_w{width}_d{depth}_{dataset}_unscaled.pth",
                )

                # store the loss barriers
                barriers_unscaled_perm[(i, j)] = plot_results(
                    pi_model,
                    models[j],
                    train_loader,
                    test_loader,
                    name=f"{dataset}_model{i}_model{j}_unscaled_perm",
                    save=True,
                )

                # rescale the models
                modeli_rescaled = normalize_weights(models[i])
                modelj_rescaled = normalize_weights(models[j])

                # store the loss barriers
                barriers_rescaled_naive[(i, j)] = plot_results(
                    modeli_rescaled,
                    modelj_rescaled,
                    train_loader,
                    test_loader,
                    name=f"{dataset}_model{i}_model{j}_rescaled_naive",
                    save=False,
                )

                # align the rescaled models
                pi_model = permute_align(
                    modeli_rescaled,
                    modelj_rescaled,
                    train_loader,
                    epochs=20,
                    device=device,
                )
                torch.save(
                    pi_model,
                    f"models/pi_model{i}_model{j}_w{width}_d{depth}_{dataset}_rescaled.pth",
                )

                # store the loss barriers
                barriers_rescaled_perm[(i, j)] = plot_results(
                    pi_model,
                    modelj_rescaled,
                    train_loader,
                    test_loader,
                    name=f"{dataset}_model{i}_model{j}_rescaled_perm",
                    save=True,
                )

        # visualize loss statistics and save plot
        plot_loss_stats(
            barriers_unscaled_naive,
            barriers_unscaled_perm,
            barriers_rescaled_naive,
            barriers_rescaled_perm,
            num_models=num_models,
            dataset=dataset,
        )

elif experiment == "barriers":
    # # given a np array of barriers, plot the heatmap with plotly
    # # if you hover over a cell, it will show the barrier value
    # # if you click on a cell, it will open the corresponding plot

    # # config
    # dataset = "MOONS"
    # scale = "Rescaled"
    # perm = "Perm"
    # data = "train"

    # # load the barriers
    # barriers = np.load(f"barriers/{dataset}_{scale} {perm}_{data}.npy")

    # def plot_heatmap(barriers, dataset, scale, perm, data):
    #     import plotly.graph_objects as go

    #     fig = go.Figure(
    #         data=go.Heatmap(
    #             z=barriers,
    #             x=[f"Model {i}" for i in range(barriers.shape[0])],
    #             y=[f"Model {i}" for i in range(barriers.shape[1])],
    #             hoverongaps=False,
    #         )
    #     )
    #     fig.update_layout(
    #         title=f"{dataset} {scale} {perm} {data}",
    #         xaxis_nticks=36,
    #     )

    #     # hover text goes here
    #     text = []
    #     for i in range(barriers.shape[0]):
    #         for j in range(barriers.shape[1]):
    #             text.append(f"Barrier: {barriers[i, j]}")

    #     fig.data[0].text = text
    #     fig.data[0].hovertemplate = "%{text}<extra></extra>"

    #     # click event goes here
    #     fig.data[0].on_click(
    #         lambda trace, points, state: webbrowser.open(
    #             f"plots/{dataset}_{scale} {perm}_{data}_model{points.point_inds[0]}_model{points.point_inds[1]}.html"
    #         )
    #     )

    #     fig.show()

    # # plot the heatmap
    # plot_heatmap(barriers, dataset, scale, perm, data)
    print("PENDING")


else:
    raise ValueError("Experiment must be either 'permute' or 'embed'")
