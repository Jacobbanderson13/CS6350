import numpy as np
import pandas as pd

train_data = pd.read_csv('Data/train.csv', header=None)
test_data = pd.read_csv('Data/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(w, X, y, v=None):
    linear_model = np.dot(X, w)
    likelihood = np.log(1 + np.exp(-y * linear_model))
    if v is not None:
        reg = (1 / (2 * v)) * np.sum(w**2)
        return np.mean(likelihood) + reg
    return np.mean(likelihood)

def compute_gradient(w, X, y, v=None):
    linear_model = np.dot(X, w)
    prob = sigmoid(-y * linear_model)
    grad = np.dot(X.T, prob * (-y)) / len(y)
    if v is not None:
        grad += w / v
    return grad

def stochastic_gradient_descent(X, y, v=None, T=100, gamma0=0.01, d=0.1):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    objective_values = []
    
    for epoch in range(T):
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        for i in range(n_samples):
            gamma_t = gamma0 / (1 + (gamma0 / d) * (epoch * n_samples + i))
            grad = compute_gradient(w, X[i:i+1], y[i:i+1], v)
            w -= gamma_t * grad
        
        obj_value = logistic_loss(w, X, y, v)
        objective_values.append(obj_value)
    
    return w, objective_values

def evaluate(w, X, y):
    predictions = np.sign(np.dot(X, w))
    return np.mean(predictions == y)

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
results = []

for v in variances:
    w_map, obj_values_map = stochastic_gradient_descent(X_train, y_train, v=v, gamma0=0.01, d=0.1)
    train_acc = evaluate(w_map, X_train, y_train)
    test_acc = evaluate(w_map, X_test, y_test)
    results.append((v, train_acc, test_acc))

w_ml, obj_values_ml = stochastic_gradient_descent(X_train, y_train, v=None, gamma0=0.01, d=0.1)
ml_train_acc = evaluate(w_ml, X_train, y_train)
ml_test_acc = evaluate(w_ml, X_test, y_test)

print("MAP Results (Variance, Train Accuracy, Test Accuracy):")
for result in results:
    print(f"v={result[0]}: Train Acc={result[1]:.4f}, Test Acc={result[2]:.4f}")

print(f"ML Results: Train Acc={ml_train_acc:.4f}, Test Acc={ml_test_acc:.4f}")