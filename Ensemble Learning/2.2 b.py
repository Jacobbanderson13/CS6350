#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import numpy as np
import pandas as pd
import math
from collections import Counter
from joblib import Parallel, delayed


# In[6]:


class ID3:
    def __init__(self, max_depth=None, criterion='information_gain'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = {}
        self.train_df = None
    
    def fit(self, df, target_attribute_name="y", attributes=None, depth=0):
        if depth == 0:
            self.train_df = df
        if attributes is None:
            attributes = df.columns.tolist()
            attributes.remove(target_attribute_name)
        

        if len(np.unique(df[target_attribute_name])) == 1:
            return np.unique(df[target_attribute_name])[0]
        elif len(df) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(df[target_attribute_name]).most_common(1)[0][0]
        elif len(attributes) == 0:
            return Counter(df[target_attribute_name]).most_common(1)[0][0]
        

        best_attr = self.get_best_split(df, target_attribute_name, attributes)


        tree = {best_attr: {}}
        remaining_attrs = [attr for attr in attributes if attr != best_attr]


        for val in np.unique(df[best_attr]):
            subset = df[df[best_attr] == val]
            subtree = self.fit(subset, target_attribute_name, remaining_attrs, depth + 1)
            tree[best_attr][val] = subtree
        
        self.tree = tree
        return tree

    def get_best_split(self, df, target_attribute_name, attributes):
        total_entropy = self.entropy(df, target_attribute_name)
        best_attr, best_gain = None, -float('inf')
        
        for attr in attributes:
            if self.criterion == 'information_gain':
                gain = self.information_gain(df, attr, target_attribute_name, total_entropy)
            elif self.criterion == 'gini_index':
                gain = -self.gini_index(df, attr, target_attribute_name)
            elif self.criterion == 'majority_error':
                gain = -self.majority_error(df, attr, target_attribute_name)
                
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                
        return best_attr

    def entropy(self, df, target_attribute_name):
        values, counts = np.unique(df[target_attribute_name], return_counts=True)
        entropy = sum([-counts[i]/sum(counts) * math.log2(counts[i]/sum(counts)) for i in range(len(values))])
        return entropy

    def gini_index(self, df, attribute, target_attribute_name):
        gini = 0
        for val in np.unique(df[attribute]):
            subset = df[df[attribute] == val]
            _, counts = np.unique(subset[target_attribute_name], return_counts=True)
            impurity = 1 - sum((counts[i]/sum(counts))**2 for i in range(len(counts)))
            gini += (len(subset)/len(df)) * impurity
        return gini

    def majority_error(self, df, attribute, target_attribute_name):
        error = 0
        for val in np.unique(df[attribute]):
            subset = df[df[attribute] == val]
            counts = Counter(subset[target_attribute_name])
            majority_class = counts.most_common(1)[0][1]
            error += (len(subset)/len(df)) * (1 - majority_class/len(subset))
        return error

    def information_gain(self, df, attribute, target_attribute_name, total_entropy=None):
        if total_entropy is None:
            total_entropy = self.entropy(df, target_attribute_name)
        weighted_entropy = 0
        for val in np.unique(df[attribute]):
            subset = df[df[attribute] == val]
            weighted_entropy += (len(subset)/len(df)) * self.entropy(subset, target_attribute_name)
        return total_entropy - weighted_entropy


    def predict_instance(self, instance, tree):
        if not isinstance(tree, dict):
            return tree
        else:
            attr = next(iter(tree))
            if attr in instance:
                val = instance[attr]
                if val in tree[attr]:
                    subtree = tree[attr][val]
                    return self.predict_instance(instance, subtree)
                else:
                    return Counter(self.train_df['y']).most_common(1)[0][0]
            else:
                return Counter(self.train_df['y']).most_common(1)[0][0]
    
    def predict(self, df):
        return df.apply(lambda x: self.predict_instance(x, self.tree), axis=1)

    def preprocess_data(self, train, test, handle_unknown='as_value'):
        train = train.copy()
        test = test.copy()

        train['y'] = train['y'].map({'yes': 1, 'no': 0})
        test['y'] = test['y'].map({'yes': 1, 'no': 0})
        

        numerical_attrs = train.select_dtypes(include=[np.number]).columns.tolist()
        numerical_attrs.remove('y')
        categorical_attrs = train.select_dtypes(exclude=[np.number]).columns.tolist()
        medians = {}
        for attr in numerical_attrs:
            med = train[attr].median()
            medians[attr] = med
            train[attr] = (train[attr] >= med).astype(str)
            test[attr] = (test[attr] >= med).astype(str)
            
            
        if handle_unknown == 'as_value':
            pass
        elif handle_unknown == 'most_common':
            for attr in categorical_attrs:
                most_common = train[train[attr] != 'unknown'][attr].mode()[0]
                train.loc[train[attr] == 'unknown', attr] = most_common
                test.loc[test[attr] == 'unknown', attr] = most_common
        else:
            raise ValueError("handle_unknown must be 'as_value' or 'most_common'")
        
        categorical_attrs = categorical_attrs + numerical_attrs
        

        combined_df = pd.concat([train, test], sort=False)
        

        combined_df = pd.get_dummies(combined_df, columns=categorical_attrs)
        
        train = combined_df.iloc[:len(train), :].copy()
        test = combined_df.iloc[len(train):, :].copy()
        
        return train, test
    
    def compute_error(self, y_true, y_pred):
        return np.mean(y_true != y_pred)


class BaggedTrees:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def bootstrap_sample(self, df):
        n_samples = len(df)
        return df.sample(n=n_samples, replace=True)

    def fit(self, df, target_attribute_name):
        self.trees = Parallel(n_jobs=-1)(delayed(self.build_tree)(df, target_attribute_name) for _ in range(self.n_trees))

    def build_tree(self, df, target_attribute_name):
        tree = ID3()
        bootstrap_df = self.bootstrap_sample(df)
        tree.fit(bootstrap_df, target_attribute_name)
        return tree

    def predict(self, df):
        tree_predictions = np.array([tree.predict(df) for tree in self.trees])
        majority_votes = pd.DataFrame(tree_predictions).mode(axis=0).iloc[0]
        return majority_votes

    def compute_error(self, y_true, y_pred):
        return np.mean(y_true != y_pred)


# In[14]:


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

def compute_error(self, y_true, y_pred):
    return np.mean(y_true != y_pred)


# In[16]:


train_df = pd.read_csv("bank/train.csv", header=None)
test_df = pd.read_csv("bank/test.csv", header=None)

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

train_df.columns = column_names
test_df.columns = column_names

train_processed, test_processed = preprocess_data(train_df, test_df)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'n_trees_list = list(range(1, 501))\n\ntrain_errors = []\ntest_errors = []\n\nfor n_trees in n_trees_list:\n    bagging_model = BaggedTrees(n_trees=n_trees)\n    bagging_model.fit(train_processed, target_attribute_name="y")\n\n    train_pred = bagging_model.predict(train_processed)\n    test_pred = bagging_model.predict(test_processed)\n\n    train_error = bagging_model.compute_error(train_processed["y"], train_pred)\n    test_error = bagging_model.compute_error(test_processed["y"], test_pred)\n\n    train_errors.append(train_error)\n    test_errors.append(test_error)\n\n\n\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\nplt.plot(n_trees_list, train_errors, label="Training Error")\nplt.plot(n_trees_list, test_errors, label="Test Error")\nplt.xlabel("Number of Trees")\nplt.ylabel("Error")\nplt.title("Training and Test Errors vs Number of Bagged Trees")\nplt.legend()\nplt.grid(True)\nplt.show()\n')


# In[ ]:




