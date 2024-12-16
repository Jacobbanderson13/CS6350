import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('Data/train.csv', header=None)
test_data = pd.read_csv('Data/test.csv', header=None)

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(len(self.layer_sizes) - 1)]
        self.biases = [np.zeros((1, size)) for size in self.layer_sizes[1:]]
    
    def forward(self, X):
        activations = [X]
        pre_activations = []
        for W, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], W) + b
            pre_activations.append(z)
            activations.append(sigmoid(z))
        return activations, pre_activations

    def backpropagation(self, X, y):
        activations, pre_activations = self.forward(X)
        gradients_W = [np.zeros_like(W) for W in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        delta = (activations[-1] - y) * sigmoid_derivative(pre_activations[-1])
        
        for i in reversed(range(len(self.weights))):
            gradients_W[i] = np.dot(activations[i].T, delta)
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(pre_activations[i-1])
        
        return gradients_W, gradients_b
    
    def update_weights(self, gradients_W, gradients_b, lr):
        self.weights = [W - lr * dW for W, dW in zip(self.weights, gradients_W)]
        self.biases = [b - lr * db for b, db in zip(self.biases, gradients_b)]

    def predict(self, X):
        activations, _ = self.forward(X)
        return (activations[-1] >= 0.5).astype(int)

    def compute_loss(self, y_pred, y_true):
        return 0.5 * np.mean((y_pred - y_true) ** 2)

def train_neural_network(X_train, y_train, X_test, y_test, hidden_layer_sizes, epochs, lr, lr_decay):
    input_size = X_train.shape[1]
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_layer_sizes, output_size)
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        
        for i in range(len(X_train)):
            gradients_W, gradients_b = nn.backpropagation(X_train[i:i+1], y_train[i:i+1])
            nn.update_weights(gradients_W, gradients_b, lr)
        
        lr /= (1 + lr_decay * epoch)
        
        train_loss = nn.compute_loss(nn.forward(X_train)[0][-1], y_train)
        test_loss = nn.compute_loss(nn.forward(X_test)[0][-1], y_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    return nn, train_losses, test_losses

hidden_layer_widths = [5, 10, 25, 50, 100]
epochs = 100
lr = 0.01
lr_decay = 0.001

results = []
for width in hidden_layer_widths:
    print(f"Training with hidden layer width: {width}")
    nn, train_losses, test_losses = train_neural_network(X_train, y_train, X_test, y_test, [width, width], epochs, lr, lr_decay)
    
    y_train_pred = nn.predict(X_train)
    y_test_pred = nn.predict(X_test)
    train_error = np.mean(y_train_pred != y_train)
    test_error = np.mean(y_test_pred != y_test)
    
    results.append((width, train_error, test_error))
    
    plt.plot(train_losses, label=f"Train Loss (width={width})")
    plt.plot(test_losses, label=f"Test Loss (width={width})")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss Convergence")
plt.show()

for width, train_error, test_error in results:
    print(f"Width: {width}, Train Error: {train_error:.10f}, Test Error: {test_error:.10f}")