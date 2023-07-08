import torch
import torch.nn as nn

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the model
class FCNet(nn.Module):
    def __init__(
        self, input_size: int, width: int, depth: int, output_size: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, width))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, output_size))

    # use ReLU activation for all layers except the last one
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:  # type: ignore # holds submodules in a list
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class FCNet_multiclass(nn.Module):
    def __init__(
        self, input_size: int, width: int, depth: int, output_size: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, width))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, output_size))

    # use ReLU activation for all layers except the last one
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the input
        x = x.view(x.shape[0], -1)
        for layer in self.layers[:-1]:  # type: ignore # holds submodules in a list
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


# Define the training loop
def train(model, train_loader, epochs=100, lr=0.001, model_name="model"):
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train the model
    model.to(device)
    model.train()
    loss = 0.0
    for epoch in range(epochs):
        for x, y in train_loader:
            # model has 1 output
            y = y.unsqueeze(1)
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))

            # Backward pass
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # save the model
    torch.save(model.state_dict(), f"models/{model_name}.pth")


def train_multiclass(model, train_loader, epochs=100, lr=0.001, model_name="model"):
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train the model
    model.to(device)
    model.train()
    loss = 0.0
    for epoch in range(epochs):
        for x, y in train_loader:
            # forward pass
            optimizer.zero_grad()
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))

            # backward pass
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    # save the model
    torch.save(model.state_dict(), f"models/{model_name}.pth")
