import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

train_data = pd.read_csv('Data/train.csv', header=None)
test_data = pd.read_csv('Data/test.csv', header=None)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, activation):
        super(NeuralNet, self).__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size))
        if activation == "tanh":
            layers[-1].weight.data = nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(nn.Tanh())
        elif activation == "relu":
            layers[-1].weight.data = nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
            layers.append(nn.ReLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation == "tanh":
                layers[-1].weight.data = nn.init.xavier_uniform_(layers[-1].weight)
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers[-1].weight.data = nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_and_evaluate(depth, width, activation):
    model = NeuralNet(X_train.shape[1], width, 1, depth, activation)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)
        train_error = mean_squared_error(y_train.numpy(), train_pred.numpy())
        test_error = mean_squared_error(y_test.numpy(), test_pred.numpy())

    return train_error, test_error

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = ["tanh", "relu"]

results = []

for activation in activations:
    for depth in depths:
        for width in widths:
            train_error, test_error = train_and_evaluate(depth, width, activation)
            results.append({
                "Activation": activation,
                "Depth": depth,
                "Width": width,
                "Train Error": train_error,
                "Test Error": test_error
            })

results_df = pd.DataFrame(results)
print(results_df)