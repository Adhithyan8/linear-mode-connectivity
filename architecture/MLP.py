import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the model
class MLP(nn.Module):
    def __init__(
        self,
        input_size=2,
        width=64,
        depth=1,
        output_size=1,
        layer_norm=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, width))
        if layer_norm:
            self.layers.append(nn.LayerNorm(width))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
            if layer_norm:
                self.layers.append(nn.LayerNorm(width))
        self.layers.append(nn.Linear(width, output_size))

    # use ReLU for all except the last
    def forward(self, x):  # flatten
        x = x.view(x.shape[0], -1)
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


# Define the training loop
def train(
    model,
    loader,
    criterion,
    optimizer,
    lr_scheduler=None,
    scheduler="batchwise",
    epochs=100,
    model_name="model",
):
    model.to(device)
    model.train()

    loss = 0.0
    for _ in range(epochs):
        for x, y in loader:
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                y = y.unsqueeze(1)
            # forward pass
            optimizer.zero_grad()
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))
            # backward pass
            loss.backward()
            optimizer.step()
            if scheduler == "batchwise":
                lr_scheduler.step()
        if scheduler == "epochwise":
            lr_scheduler.step()

    # save the model weights
    torch.save(model.state_dict(), f"models/{model_name}.pth")
