import numpy as np
import torch

from architecture.MLP import FCNet, train
from permute import permute_align
from utils import (
    evaluate,
    get_data,
    interpolation_losses,
    normalize_weights,
    loss_barrier,
)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# datasets
datasets = "gaussian"
n_samples = 512
n_widths = [3, 4, 5, 6, 12]


"""
for n_width in n_widths:
    # Load the data
    train_loader, test_loader = get_data(name=datasets, n_samples=n_samples)

    # config
    num_models = 50
    width = n_width
    depth = 1
    epochs = 100

    # data structure to store losses and accuracies
    logs = np.zeros((num_models, 4))

    # data structure to store interpolation losses
    int_losses = np.zeros((num_models, num_models, 11))

    # data structure to store loss barriers
    loss_barriers = np.zeros((num_models, num_models))

    # Define and train many models
    models = []
    for i in range(num_models):
        model = FCNet(input_size=2, width=width, depth=depth, output_size=1)
        train(
            model,
            train_loader,
            epochs=epochs,
            lr=0.001,
            model_name=f"{datasets}/model_s{n_samples}_w{width}_d{depth}_{i}",
        )
        models.append(model)

        # compute train loss, test loss, train accuracy, test accuracy
        # log the results
        model.eval()

        train_loss, train_acc = evaluate(model, train_loader)
        test_loss, test_acc = evaluate(model, test_loader)

        logs[i, 0] = train_loss
        logs[i, 1] = test_loss
        logs[i, 2] = train_acc
        logs[i, 3] = test_acc

        # save the log
        np.save(
            f"logs/sigmoid/{datasets}/logs_s{n_samples}_w{width}_d{depth}_{i}", logs
        )

    # pick index of model with lowest train loss as reference
    reference_model_idx = np.argmin(logs[:, 0])

    # normalize weights of reference model
    models[reference_model_idx] = normalize_weights(models[reference_model_idx])

    # align all other models to this reference model
    for i in range(num_models):
        if i == reference_model_idx:
            continue
        models[i] = normalize_weights(models[i])

        models[reference_model_idx].eval().to(device)
        models[i].eval().to(device)

        models[i] = permute_align(
            models[i],
            models[reference_model_idx],
            train_loader,
            epochs=20,
            device=device,
        )

    # compute interpolation loss for each pair of models
    # log the results
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            if i > j:
                int_losses[i, j, :] = int_losses[j, i, :]
                loss_barriers[i, j] = loss_barriers[j, i]
                continue
            if i < j:
                int_losses[i, j, :] = interpolation_losses(
                    models[i], models[j], train_loader
                )
                loss_barriers[i, j] = loss_barrier(int_losses[i, j, :])

    # save the interpolation losses
    np.save(
        f"logs/sigmoid/{datasets}/interpolation_losses_s{n_samples}_w{width}_d{depth}",
        int_losses,
    )

    # save the loss barriers
    np.save(
        f"logs/sigmoid/{datasets}/loss_barriers_s{n_samples}_w{width}_d{depth}",
        loss_barriers,
    )
"""

for n_width in n_widths:
    # Load the data
    train_loader, test_loader = get_data(name=datasets, n_samples=n_samples)

    # config
    num_models = 50
    width = n_width
    depth = 1
    epochs = 100

    # data structure to store losses and accuracies
    logs = np.zeros((num_models, 4))

    # data structure to store interpolation losses
    int_losses = np.zeros((num_models, num_models, 11))

    # data structure to store loss barriers
    loss_barriers = np.zeros((num_models, num_models))

    # Define and load models
    models = []
    for i in range(num_models):
        model = FCNet(input_size=2, width=width, depth=depth, output_size=1)
        model.load_state_dict(
            torch.load(
                f"models/sigmoid/{datasets}/model_s{n_samples}_w{width}_d{depth}_{i}"
            )
        )
        models.append(model)

        # compute train loss, test loss, train accuracy, test accuracy
        # log the results
        model.eval()

        train_loss, train_acc = evaluate(model, train_loader)
        test_loss, test_acc = evaluate(model, test_loader)

        logs[i, 0] = train_loss
        logs[i, 1] = test_loss
        logs[i, 2] = train_acc
        logs[i, 3] = test_acc

    # compute interpolation loss for each pair of models
    # log the results
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            if i > j:
                int_losses[i, j, :] = int_losses[j, i, :]
                loss_barriers[i, j] = loss_barriers[j, i]
                continue
            if i < j:
                int_losses[i, j, :] = interpolation_losses(
                    models[i], models[j], train_loader
                )
                loss_barriers[i, j] = loss_barrier(int_losses[i, j, :])

    # save the interpolation losses
    np.save(
        f"logs/sigmoid/{datasets}/naive_interpolation_losses_s{n_samples}_w{width}_d{depth}",
        int_losses,
    )

    # save the loss barriers
    np.save(
        f"logs/sigmoid/{datasets}/naive_loss_barriers_s{n_samples}_w{width}_d{depth}",
        loss_barriers,
    )
