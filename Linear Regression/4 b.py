#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


def compute_cost(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def stochastic_gradient_descent(X, y, r, tolerance=1e-6, max_iterations=10000):
    m, n = X.shape
    w = np.zeros(n)
    cost_history = []
    
    for iteration in range(max_iterations):
        idx = np.random.randint(m)
        X_i = X[idx:idx+1]
        y_i = y[idx:idx+1]
        
        prediction = X_i.dot(w)
        error = prediction - y_i
        gradient = X_i.T.dot(error)
        w = w - r * gradient
        
        cost = compute_cost(X, y, w)
        cost_history.append(cost)
        
        if iteration > 1 and np.abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
    
    return w, cost_history


# In[8]:


train_df = pd.read_csv("concrete/train.csv", header=None)
test_df = pd.read_csv("concrete/test.csv", header=None)

column_names = ['Cement',
'Slag',
'Fly ash',
'Water',
'SP',
'Coarse Aggr',
'Fine Aggr',
'y']

train_df.columns = column_names
test_df.columns = column_names


# In[14]:


train_y = train_df['y'].values

train_x = train_df.drop(columns=['y']).values 

test_y = test_df['y'].values
test_x = test_df.drop(columns=['y']).values


# In[16]:


X_b = np.c_[np.ones((53, 1)), train_x]
y = train_y


# In[18]:


# Try different learning rates
learning_rates = [0.1, 0.01, 0.001]
tolerance = 1e-6
for r in learning_rates:
    print(f"Trying learning rate: {r}")
    w, cost_history = stochastic_gradient_descent(X_b, y, r, tolerance=tolerance)
    print(f"Learned weights: {w}")
    print(f"Final cost: {cost_history[-1]}\n")
    
    # Plot cost history
    plt.plot(cost_history, label=f"Learning rate: {r}")

plt.title("Cost function over stochastic updates")
plt.xlabel("Number of updates")
plt.ylabel("Cost")
plt.legend()
plt.show()


# In[ ]:




