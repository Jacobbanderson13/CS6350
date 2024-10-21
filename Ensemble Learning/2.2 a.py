#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
from collections import Counter


# In[ ]:


class DecisionStump:
    def __init__(self, criterion='information_gain'):
        self.criterion = criterion
        self.stump = None
        self.train_df = None
    
    def fit(self, df, target_attribute_name="y", attributes=None, weights=None):
        self.train_df = df.copy()
        
        if weights is None:
            weights = np.ones(len(df)) / len(df)
            
        if attributes is None:
            attributes = df.columns.tolist()
            attributes.remove(target_attribute_name)
        
        if len(np.unique(df[target_attribute_name])) == 1:
            self.stump = np.unique(df[target_attribute_name])[0]
            return
        
        best_attr = self.get_best_attribute(df, target_attribute_name, attributes, weights)
        
        self.stump = {best_attr: {}}
        for value in np.unique(df[best_attr]):
            subset = df[df[best_attr] == value]
            subset_weights = weights[df[best_attr] == value]
            
            most_common_label = self.get_weighted_most_common_label(subset, target_attribute_name, subset_weights)
            self.stump[best_attr][value] = most_common_label
    
    def get_weighted_most_common_label(self, df, target_attribute_name, weights):
        label_counts = Counter()
        for label, weight in zip(df[target_attribute_name], weights):
            label_counts[label] += weight
        return label_counts.most_common(1)[0][0]
    
    def get_best_attribute(self, df, target_attribute_name, attributes, weights):
        best_gain = -1
        best_attr = None
        for attr in attributes:
            gain = self.information_gain(df, target_attribute_name, attr, weights)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr
    
    def information_gain(self, df, target_attribute_name, attribute, weights):
        total_entropy = self.entropy(df[target_attribute_name], weights)
        values, counts = np.unique(df[attribute], return_counts=True)
        
        weighted_avg_entropy = 0
        for value, count in zip(values, counts):
            subset = df[df[attribute] == value]
            subset_weights = weights[df[attribute] == value]
            weighted_avg_entropy += (np.sum(subset_weights) / np.sum(weights)) * self.entropy(subset[target_attribute_name], subset_weights)
        
        return total_entropy - weighted_avg_entropy
    
    def entropy(self, labels, weights):
        label_counts = Counter()
        for label, weight in zip(labels, weights):
            label_counts[label] += weight
        
        entropy = 0
        total_weight = np.sum(weights)
        for label in label_counts:
            p_label = label_counts[label] / total_weight
            if p_label > 0:
                entropy -= p_label * math.log2(p_label)
        
        return entropy
    
    def predict(self, X):
        # if self.stump is None:
        #     raise Exception("stump not fit")
        
        predictions = []
        for _, row in X.iterrows():
            attr = list(self.stump.keys())[0]
            value = row[attr]
            if value in self.stump[attr]:
                predictions.append(self.stump[attr][value])
            else:
                predictions.append(Counter(self.train_df['y']).most_common(1)[0][0])
        return np.array(predictions)


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, df, target_attribute_name="y", attributes=None):
        n_samples = len(df)
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(df, target_attribute_name, attributes, weights)
            predictions = stump.predict(df)
            
            incorrect = predictions != df[target_attribute_name]
            error = np.dot(weights, incorrect) / np.sum(weights)
            
            alpha = 0.5 * math.log((1 - error) / (error + 1e-10))
            self.alphas.append(alpha)
            self.stumps.append(stump)
            
            weights *= np.exp(-alpha * (predictions == df[target_attribute_name]) * 2 - 1)
            weights /= np.sum(weights)
    
    def predict(self, X):
        stump_preds = np.array([alpha * stump.predict(X) for stump, alpha in zip(self.stumps, self.alphas)])
        final_preds = np.sign(np.sum(stump_preds, axis=0))
        return final_preds


# In[ ]:


def preprocess_data(train_df, test_df, handle_unknown='as_value'):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['y'] = train_df['y'].map({'yes': 1, 'no': -1})
    test_df['y'] = test_df['y'].map({'yes': 1, 'no': -1})
    
    numerical_attrs = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    
    medians = {}
    for attr in numerical_attrs:
        med = train_df[attr].median()
        medians[attr] = med
        train_df[attr] = (train_df[attr] >= med).astype(str)
        test_df[attr] = (test_df[attr] >= med).astype(str)
    
    if handle_unknown == 'as_value':
        pass
    elif handle_unknown == 'most_common':
        categorical_attrs = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        for attr in categorical_attrs:
            most_common = train_df[train_df[attr] != 'unknown'][attr].mode()[0]
            train_df.loc[train_df[attr] == 'unknown', attr] = most_common
            test_df.loc[test_df[attr] == 'unknown', attr] = most_common
    else:
        raise ValueError("handle_unknown must be 'as_value' or 'most_common'")
    
    categorical_attrs = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'] + numerical_attrs
    
    combined_df = pd.concat([train_df, test_df], sort=False)
    
    combined_df = pd.get_dummies(combined_df, columns=categorical_attrs)
    
    train_df = combined_df.iloc[:len(train_df), :].copy()
    test_df = combined_df.iloc[len(train_df):, :].copy()
    
    return train_df, test_df


# In[ ]:


train_df = pd.read_csv("bank/train.csv", header=None)
test_df = pd.read_csv("bank/test.csv", header=None)

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

train_df.columns = column_names
test_df.columns = column_names

train_processed, test_processed = preprocess_data(train_df, test_df)


# In[ ]:


def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


# In[ ]:


import matplotlib.pyplot as plt

train_errors = []
test_errors = []

n = 501
for t in range(1, n):
    adaboost = AdaBoost(n_estimators=t)
    adaboost.fit(train_processed, target_attribute_name="y")

    train_pred = adaboost.predict(train_processed)
    test_pred = adaboost.predict(test_processed)

    train_error = compute_error(train_processed["y"], train_pred)
    test_error = compute_error(test_processed["y"], test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)



# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(range(1, n), train_errors, label="Training Error")
plt.plot(range(1, n), test_errors, label="Test Error")
plt.xlabel("Number of Iterations (T)")
plt.ylabel("Error")
plt.title("Training and Test Errors over AdaBoost Iterations")
plt.legend()
plt.show()


# In[ ]:




