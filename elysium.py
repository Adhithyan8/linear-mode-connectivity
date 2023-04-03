import numpy as np
import torch
import os

from architecture.MLP import FCNet, train
from permute import permute_align
from utils import (
    get_data,
    normalize_weights,
    save_loss_stats,
    plot_results,
    interactive_heatmap,
)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment = "permute"  # "permute" or "embed"

# experiment 1: align models and plot loss barriers
if experiment == "permute":
    # datasets
    datasets = ["moons", "gaussian"]
    n_samples = [128, 256, 512]
    n_widths = [4, 32, 1024]

    for dataset in datasets:
        for n_sample in n_samples:
            for n_width in n_widths:
                # Load the data
                train_loader, test_loader = get_data(name=dataset, n_samples=n_sample)

                # config
                num_models = 50
                width = n_width
                depth = 1
                epochs = 100

                # Define and train many models
                models = []
                for i in range(num_models):
                    model = FCNet(input_size=2, width=width, depth=depth, output_size=1)
                    train(
                        model,
                        train_loader,
                        epochs=epochs,
                        lr=0.001,
                        model_name=f"{dataset}/model_s{n_sample}_w{width}_d{depth}_{i}",
                    )
                    models.append(model)

                # or load the models from disk
                models = []
                for i in range(num_models):
                    model = FCNet(input_size=2, width=width, depth=depth, output_size=1)
                    model.load_state_dict(
                        torch.load(
                            f"models/sigmoid/{dataset}/model_s{n_sample}_w{width}_d{depth}_{i}.pth"
                        )
                    )
                    models.append(model)

                barriers_perm = dict()
                barriers_scale_perm = dict()

                for i in range(num_models):
                    for j in range(i + 1, num_models):
                        models[i].eval().to(device)
                        models[j].eval().to(device)

                        # align the models
                        pi_model = permute_align(
                            models[i], models[j], train_loader, epochs=20, device=device
                        )
                        torch.save(
                            pi_model,
                            f"models/sigmoid/{dataset}/pi_model{i}_model{j}_s{n_sample}_w{width}_d{depth}.pth",
                        )

                        # store the loss barriers
                        barriers_perm[(i, j)] = plot_results(
                            pi_model,
                            models[j],
                            train_loader,
                            test_loader,
                            name=f"{dataset}_model{i}_model{j}_unscaled_perm",
                            save=False,
                        )

                        # rescale the models
                        modeli_rescaled = normalize_weights(models[i])
                        modelj_rescaled = normalize_weights(models[j])

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
                            f"models/sigmoid/{dataset}/pi_model{i}_model{j}_s{n_sample}_w{width}_d{depth}_scale.pth",
                        )

                        # store the loss barriers
                        barriers_scale_perm[(i, j)] = plot_results(
                            pi_model,
                            modelj_rescaled,
                            train_loader,
                            test_loader,
                            name=f"{dataset}_model{i}_model{j}_rescaled_perm",
                            save=False,
                        )

                        # save loss statistics
                        save_loss_stats(
                            barriers_perm,
                            barriers_scale_perm,
                            num_models=num_models,
                            name=f"{dataset}/s{n_sample}_w{width}",
                        )

# experiment 2: using loss barriers, embed models in 2D space
elif experiment == "embed":
    # get path of D:\03_KTH\Thesis\LMC\barriers\softmax\moons\scale_perm_train.npy
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "barriers", "softmax", "moons", "scale_perm_train.npy")

    # plot interactive heatmap
    interactive_heatmap(path)

else:
    raise ValueError("Experiment must be either 'permute' or 'embed'")
