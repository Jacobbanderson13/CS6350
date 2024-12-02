import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize


train_data = pd.read_csv('data/train.csv', header=None)
test_data = pd.read_csv('data/test.csv', header=None)


X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values


y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

def svm_loss(w, X, y, C):
    n = X.shape[0]
    hinge_loss = np.maximum(0, 1 - y * (np.dot(X, w)))
    return 0.5 * np.dot(w, w) + C * np.sum(hinge_loss) / n

def svm_subgradient(w, X, y, C):
    n = X.shape[0]
    margins = y * (np.dot(X, w))
    subgrad = np.where(margins < 1, -y[:, None] * X, 0)
    return w + C * np.sum(subgrad, axis=0) / n

def stochastic_subgradient_svm(X, y, C, T, lr_schedule, gamma_0, a=None):
    n, d = X.shape
    w = np.zeros(d)
    losses = []

    for epoch in range(T):
        indices = np.random.permutation(n)
        X, y = X[indices], y[indices]

        for t in range(n):
            step = epoch * n + t + 1
            gamma_t = gamma_0 / (1 + gamma_0 / a * step) if a else gamma_0 / (1 + step)
            
            margin = y[t] * np.dot(X[t], w)
            if margin < 1:
                w = (1 - gamma_t) * w + gamma_t * C * y[t] * X[t]
            else:
                w = (1 - gamma_t) * w

        losses.append(svm_loss(w, X, y, C))

    return w, losses

def calculate_error(w, X, y):
    predictions = np.sign(np.dot(X, w))
    return np.mean(predictions != y)

C_values = [100 / 873, 500 / 873, 700 / 873]
T = 100  
gamma_0 = 0.1
a = 1  

results = {}

for C in C_values:
    w1, losses1 = stochastic_subgradient_svm(X_train, y_train, C, T, "schedule_1", gamma_0, a)
    w2, losses2 = stochastic_subgradient_svm(X_train, y_train, C, T, "schedule_2", gamma_0)
    
    results[C] = {
        "w1": w1,
        "losses1": losses1,
        "w2": w2,
        "losses2": losses2,
    }



evaluation_results = {}

for C in C_values:
    result = results[C]
    w1, w2 = result["w1"], result["w2"]
    
    train_error1 = calculate_error(w1, X_train, y_train)
    test_error1 = calculate_error(w1, X_test, y_test)
    
    train_error2 = calculate_error(w2, X_train, y_train)
    test_error2 = calculate_error(w2, X_test, y_test)
    
    evaluation_results[C] = {
        "train_error1": train_error1,
        "test_error1": test_error1,
        "train_error2": train_error2,
        "test_error2": test_error2,
    }

print(evaluation_results)

print('Next Step: Can be slow')

def dual_svm_objective(alpha, X, y, C):
    n = X.shape[0]
    alpha = alpha.reshape(-1, 1)
    y = y.reshape(-1, 1)
    gram_matrix = np.dot(y * X, (y * X).T)
    return 0.5 * np.dot(alpha.T, np.dot(gram_matrix, alpha)) - np.sum(alpha)

def dual_svm_constraint(alpha, y):
    return np.dot(alpha, y)

def solve_dual_svm(X, y, C):
    n = X.shape[0]
    bounds = [(0, C) for _ in range(n)]
    initial_alpha = np.zeros(n)
    constraints = {'type': 'eq', 'fun': dual_svm_constraint, 'args': (y,)}
    
    result = minimize(
        fun=dual_svm_objective,
        x0=initial_alpha,
        args=(X, y, C),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

def recover_weights_bias(alpha, X, y, C):
    support_vectors = alpha > 1e-6 
    w = np.sum((alpha[support_vectors] * y[support_vectors])[:, None] * X[support_vectors], axis=0)
    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))
    return w, b, np.sum(support_vectors)

dual_results = {}

for C in C_values:
    alpha = solve_dual_svm(X_train, y_train, C)
    w, b, num_support_vectors = recover_weights_bias(alpha, X_train, y_train, C)
    
    dual_results[C] = {
        "w": w,
        "b": b,
        "num_support_vectors": num_support_vectors,
    }

dual_results_df = pd.DataFrame(dual_results).T

print(dual_results_df)
