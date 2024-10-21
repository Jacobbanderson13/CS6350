#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import math
from collections import Counter
from copy import deepcopy

class ID3:
    def __init__(self, max_depth=None, criterion='information_gain', feature_subset_size=None):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = {}
        self.feature_subset_size = feature_subset_size

    def fit(self, df, target_attribute_name="y", attributes=None, depth=0):
        if attributes is None:
            attributes = df.columns.tolist()
            attributes.remove(target_attribute_name)

        if len(np.unique(df[target_attribute_name])) == 1:
            return np.unique(df[target_attribute_name])[0]
        elif len(df) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(df[target_attribute_name]).most_common(1)[0][0]
        elif len(attributes) == 0:
            return Counter(df[target_attribute_name]).most_common(1)[0][0]

        if self.feature_subset_size and len(attributes) > self.feature_subset_size:
            random_attributes = np.random.choice(attributes, self.feature_subset_size, replace=False)
        else:
            random_attributes = attributes 

        best_attr = self.get_best_split(df, target_attribute_name, random_attributes)

        tree = {best_attr: {}}
        remaining_attrs = [attr for attr in attributes if attr != best_attr]

        for val in np.unique(df[best_attr]):
            subset = df[df[best_attr] == val]
            subtree = self.fit(subset, target_attribute_name, remaining_attrs, depth + 1)
            tree[best_attr][val] = subtree

        self.tree = tree
        return tree

    def get_best_split(self, df, target_attribute_name, attributes):
        best_attr = None
        best_gain = -float('inf')

        for attr in attributes:
            gain = self.information_gain(df, attr, target_attribute_name)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr

    def entropy(self, df, target_attribute_name):
        values, counts = np.unique(df[target_attribute_name], return_counts=True)
        entropy = -sum((counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(values)))
        return entropy

    def information_gain(self, df, attr, target_attribute_name):
        total_entropy = self.entropy(df, target_attribute_name)
        values, counts = np.unique(df[attr], return_counts=True)

        weighted_entropy = sum((counts[i] / np.sum(counts)) * self.entropy(df[df[attr] == values[i]], target_attribute_name) for i in range(len(values)))
        info_gain = total_entropy - weighted_entropy
        return info_gain

    def predict_instance(self, instance, tree=None):
        if tree is None:
            tree = self.tree
        for attr, branches in tree.items():
            value = instance[attr]
            if value in branches:
                result = branches[value]
                if isinstance(result, dict):
                    return self.predict_instance(instance, result)
                else:
                    return result
        return None

    def predict(self, df):
        predictions = df.apply(lambda row: self.predict_instance(row), axis=1)
        return predictions

class RandomForest:
    def __init__(self, n_trees, feature_subset_size):
        self.n_trees = n_trees
        self.feature_subset_size = feature_subset_size
        self.trees = []

    def bootstrap_sample(self, df):
        n_samples = len(df)
        return df.sample(n=n_samples, replace=True)

    def fit(self, df, target_attribute_name):
        for _ in range(self.n_trees):
            tree = ID3(feature_subset_size=self.feature_subset_size)
            bootstrap_df = self.bootstrap_sample(df)
            tree.fit(bootstrap_df, target_attribute_name)
            self.trees.append(tree)

    def predict(self, df):
        tree_predictions = np.array([tree.predict(df) for tree in self.trees])
        majority_votes = pd.DataFrame(tree_predictions).mode(axis=0).iloc[0]
        return majority_votes

    def compute_error(self, y_true, y_pred):
        return np.mean(y_true != y_pred)


# In[12]:


def preprocess_data(train_df, test_df, handle_unknown='as_value'):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['y'] = train_df['y'].map({'yes': 1, 'no': 0})
    test_df['y'] = test_df['y'].map({'yes': 1, 'no': 0})
    
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

def compute_error(self, y_true, y_pred):
    return np.mean(y_true != y_pred)


# In[38]:


train_df = pd.read_csv("bank/train.csv", header=None)
test_df = pd.read_csv("bank/test.csv", header=None)

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

train_df.columns = column_names
test_df.columns = column_names

train_processed, test_processed = preprocess_data(train_df, test_df)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_trees_list = list(range(1, 501))\n\n\nfeature_subset_sizes = [2, 4, 6]\n\nresults = {subset_size: {"train_errors": [], "test_errors": []} for subset_size in feature_subset_sizes}\n\nfor subset_size in feature_subset_sizes:\n    for n_trees in n_trees_list:\n        random_forest_model = RandomForest(n_trees=n_trees, feature_subset_size=subset_size)\n        random_forest_model.fit(train_processed, target_attribute_name="y")\n\n        train_pred = random_forest_model.predict(train_processed)\n        test_pred = random_forest_model.predict(test_processed)\n\n        train_error = random_forest_model.compute_error(train_processed["y"], train_pred)\n        test_error = random_forest_model.compute_error(test_processed["y"], test_pred)\n\n        results[subset_size]["train_errors"].append(train_error)\n        results[subset_size]["test_errors"].append(test_error)\n\n\nimport matplotlib.pyplot as plt\nfor subset_size in feature_subset_sizes:\n    plt.figure(figsize=(10, 6))\n    plt.plot(n_trees_list, results[subset_size]["train_errors"], label=f"Train Error (Feature Subset Size = {subset_size})")\n    plt.plot(n_trees_list, results[subset_size]["test_errors"], label=f"Test Error (Feature Subset Size = {subset_size})")\n    plt.xlabel("Number of Trees")\n    plt.ylabel("Error")\n    plt.title(f"Training and Test Errors vs Number of Trees (Feature Subset Size = {subset_size})")\n    plt.legend()\n    plt.grid(True)\n    plt.show()\n')


# In[ ]:




