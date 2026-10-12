import torch
import torch.nn as nn
import numpy as np

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    input_size = X.shape[1]
    model = SingleNeuron(input_size)

    with torch.no_grad():
        model.neuron.weight.copy_(torch.tensor(initial_weights, dtype=torch.float32).view(1, -1))
        model.neuron.bias.copy_(torch.tensor([initial_bias], dtype=torch.float32))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    mse_values = []

    for _ in range(epochs):
        y_pred = model(features)
        loss = criterion(y_pred, y)
        mse_values.append(round(loss.item(), 4))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    updated_weights = [round(float(w), 4) for w in model.neuron.weight.detach().flatten()]
    updated_bias = round(float(model.neuron.bias.item()), 4)

    return updated_weights, updated_bias, mse_values



class SingleNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.neuron = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.neuron(x))