#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[75]:


def compute_cost(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def batch_gradient_descent(X, y, r, tolerance=1e-6, max_iterations=10000):
    m, n = X.shape
    w = np.zeros(n) 
    cost_history = []
    weight_diff_norms = []
    
    for iteration in range(max_iterations):
        predictions = X.dot(w)
        error = predictions - y
        gradient = (1 / m) * X.T.dot(error)
        new_w = w - r * gradient
        weight_diff_norm = np.linalg.norm(new_w - w)
        
        cost_history.append(compute_cost(X, y, new_w))
        weight_diff_norms.append(weight_diff_norm)
        
        if weight_diff_norm < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
        
        w = new_w
    
    return w, cost_history


# In[63]:


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


# In[65]:


train_y = train_df['y'].values

train_x = train_df.drop(columns=['y']).values 

test_y = test_df['y'].values
test_x = test_df.drop(columns=['y']).values


# In[67]:


X_b = np.c_[np.ones((53, 1)), train_x]
y = train_y


# In[73]:


learning_rates = [1, 0.5, 0.25, 0.125]
tolerance = 1e-6
for r in learning_rates:
    print(f"Trying learning rate: {r}")
    w, cost_history = batch_gradient_descent(X_b, y, r, tolerance=tolerance)
    print(f"Learned weights: {w}")
    print(f"Final cost: {cost_history[-1]}\n")
    
    plt.plot(cost_history, label=f"Learning rate: {r}")

plt.title("Cost function over iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.show()


# In[ ]:




