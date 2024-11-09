import numpy as np
import pandas as pd

train_data = pd.read_csv('bank-note/train.csv', header=None)
test_data = pd.read_csv('bank-note/test.csv', header=None)

def standard_perceptron(X_train, y_train, X_test, y_test, epochs, learning_rate):
    bias = 0 
    weights = np.zeros(X_train.shape[1])
    
    for epoch in range(epochs):
        for i in range(len(X_train)):
            temp = np.dot(weights, X_train[i]) + bias
            if(temp >= 0):
                pred = 1
            else:
                pred = -1 
    
            if pred != y_train[i]:
                weights += learning_rate * y_train[i] * X_train[i]
                bias += learning_rate * y_train[i]


    preds = np.array([1 if (np.dot(weights, x) + bias) >= 0 else -1 for x in X_test])
    average_error = np.mean(preds != y_test)
    
    return weights, bias, average_error

def voted_perceptron(X_train, y_train, X_test, y_test, epochs, learning_rate):
    bias = 0 
    weights = np.zeros(X_train.shape[1])
    distinct_weights = [] 
    current_count = 1 
    
    for epoch in range(epochs):
        for i in range(len(X_train)):
            temp = np.dot(weights, X_train[i]) + bias
            if(temp >= 0):
                pred = 1
            else:
                pred = -1 
    
            if pred == y_train[i]:
                current_count += 1
            else:
                distinct_weights.append((weights.copy(), bias, current_count))

                weights += learning_rate * y_train[i] * X_train[i]
                bias += learning_rate * y_train[i]

                current_count = 1
    

    distinct_weights.append((weights.copy(), bias, current_count))
    

    test_preds = []
    for x in X_test:

        vote_sum = sum(count * (1 if (np.dot(w, x) + b) >= 0 else -1) 
                       for w, b, count in distinct_weights)

        final_pred = 1 if vote_sum >= 0 else -1
        test_preds.append(final_pred)
    

    average_test_error = np.mean(np.array(test_preds) != y_test)
    
    distinct_weights_summary = [(w.tolist(), b, c) for w, b, c in distinct_weights]
    
    return distinct_weights_summary[:5], len(distinct_weights_summary), average_test_error

def average_perceptron(X_train, y_train, X_test, y_test, epochs, learning_rate):
    weights = np.zeros(X_train.shape[1])
    bias = 0 
    average_weights = np.zeros(X_train.shape[1])
    average_bias = 0
    
    for epoch in range(epochs):
        for i in range(len(X_train)):
            temp = np.dot(weights, X_train[i]) + bias
            if(temp >= 0):
                pred = 1
            else:
                pred = -1 
    
            if pred != y_train[i]:
                weights += learning_rate * y_train[i] * X_train[i]
                bias += learning_rate * y_train[i]
    
            average_weights += weights
            average_bias += bias
    

    final_average_weights = average_weights / (epochs * len(X_train))
    final_average_bias = average_bias / (epochs * len(X_train))
    
    average_test_preds = np.array([1 if (np.dot(final_average_weights, x) + final_average_bias) >= 0 else -1 for x in X_test])
    average_test_error = np.mean(average_test_preds != y_test)
    
    return final_average_weights, final_average_bias, average_test_error


X_train = train_data.iloc[:, :-1].values 
y_train = train_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1).values
X_test = test_data.iloc[:, :-1].values 
y_test = test_data.iloc[:, -1].apply(lambda x: 1 if x == 1 else -1).values



epochs = 10
learning_rate = 1

print("Standard Perceptron")
print(standard_perceptron(X_train, y_train, X_test, y_test, epochs, learning_rate))
print(" ")

print("Voted Perceptron")
print(voted_perceptron(X_train, y_train, X_test, y_test, epochs, learning_rate))
print(" ")

print("Average Perceptron")
print(average_perceptron(X_train, y_train, X_test, y_test, epochs, learning_rate))




