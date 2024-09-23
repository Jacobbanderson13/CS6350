#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from collections import Counter


# In[2]:


class ID3:
    def __init__(self, max_depth=None, criterion='information_gain'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = {}
        self.train_df = None
    
    def fit(self, df, target_attribute="y", attributes=None, depth=0):
        if depth == 0:
            self.train_df = df.copy()
        if attributes is None:
            attributes = df.columns.tolist()
            attributes.remove(target_attribute)
        
        if len(np.unique(df[target_attribute])) == 1:
            return np.unique(df[target_attribute])[0]
        elif len(df) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(df[target_attribute]).most_common(1)[0][0]
        elif len(attributes) == 0:
            return Counter(df[target_attribute]).most_common(1)[0][0]
        
        best_attr = self.get_best_split(df, target_attribute, attributes)

        tree = {best_attr: {}}
        remaining = [attr for attr in attributes if attr != best_attr]

        for val in np.unique(df[best_attr]):
            subset = df[df[best_attr] == val]
            sub = self.fit(subset, target_attribute, remaining, depth + 1)
            tree[best_attr][val] = sub
        
        self.tree = tree
        return tree

    def get_best_split(self, df, target_attribute, attributes):
        best_attr = None
        best_gain = -float('inf')

        for attr in attributes:
            if self.criterion == 'information_gain':
                gain = self.information_gain(df, attr, target_attribute)
            elif self.criterion == 'gini_index':
                gain = -self.gini_index(df, attr, target_attribute)
            elif self.criterion == 'majority_error':
                gain = -self.majority_error(df, attr, target_attribute)

            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        
        return best_attr

    def entropy(self, df, target_attribute):
        values, counts = np.unique(df[target_attribute], return_counts=True)
        entropy = sum([-counts[i]/sum(counts) * math.log2(counts[i]/sum(counts)) for i in range(len(values))])
        return entropy

    def gini_index(self, df, attribute, target_attribute):
        gini = 0
        for val in np.unique(df[attribute]):
            subset = df[df[attribute] == val]
            _, counts = np.unique(subset[target_attribute], return_counts=True)
            impurity = 1 - sum((counts[i]/sum(counts))**2 for i in range(len(counts)))
            gini += (len(subset)/len(df)) * impurity
        return gini

    def majority_error(self, df, attribute, target_attribute):
        error = 0
        for val in np.unique(df[attribute]):
            subset = df[df[attribute] == val]
            counts = Counter(subset[target_attribute])
            majority_class = counts.most_common(1)[0][1]
            error += (len(subset)/len(df)) * (1 - majority_class/len(subset))
        return error

    def information_gain(self, df, attribute, target_attribute):
        total_entropy = self.entropy(df, target_attribute)
        weighted_entropy = 0
        for val in np.unique(df[attribute]):
            subset = df[df[attribute] == val]
            weighted_entropy += (len(subset)/len(df)) * self.entropy(subset, target_attribute)
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

def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


# In[6]:


train_df = pd.read_csv("bank/train.csv", header=None)
test_df = pd.read_csv("bank/test.csv", header=None)

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

train_df.columns = column_names
test_df.columns = column_names

cases = {'Case A (unknown as value)': 'as_value', 'Case B (Use Most Common)': 'most_common'}
for case_name, handle_unknown in cases.items():
    print(case_name)
    train_processed, test_processed = preprocess_data(train_df, test_df, handle_unknown=handle_unknown)
    
    attributes = [col for col in train_processed.columns if col != 'y']
    
    criteria = ['information_gain', 'gini_index', 'majority_error']
    max_depths = range(1, 17)
    
    results = []
    for criterion in criteria:
        for depth in max_depths:
            id3 = ID3(max_depth=depth, criterion=criterion)
            id3.fit(train_processed, target_attribute='y', attributes=attributes)
            y_train_pred = id3.predict(train_processed)
            y_test_pred = id3.predict(test_processed)
            train_error = compute_error(train_processed['y'], y_train_pred)
            test_error = compute_error(test_processed['y'], y_test_pred)
            results.append({'Criterion': criterion, 'Depth': depth, 'Train Error': train_error, 'Test Error': test_error})

    results_df = pd.DataFrame(results)
    print(results_df)
    print(" ")


# In[ ]:




