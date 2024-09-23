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

    def fit(self, df, target_attribute="label", attributes=None, depth=0):
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

    def predict(self, instance):
        tree = self.tree
        while isinstance(tree, dict):
            attr = next(iter(tree))
            if instance[attr] in tree[attr]:
                tree = tree[attr][instance[attr]]
            else:
                return None
        return tree

    def predict_batch(self, df):
        return df.apply(self.predict, axis=1)


# In[3]:


train = pd.read_csv('car/train.csv', header=None)
test = pd.read_csv('car/test.csv', header=None)
cols = ['buying','maint','doors','persons','lug_boot','safety','label']
train.columns = cols
test.columns = cols


attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

model = ID3(max_depth=3, criterion='information_gain')
model.fit(train, attributes=attributes)

predictions = model.predict_batch(test)


# In[7]:


def calculate_error(predictions, actual):
    return np.mean(predictions != actual)


criteria = ['information_gain', 'majority_error', 'gini_index']

depths = range(1, 7)

results = {
    'Criterion': [],
    'Max Depth': [],
    'Train Error': [],
    'Test Error': []
}

for criterion in criteria:
    for depth in depths:
        model = ID3(max_depth=depth, criterion=criterion)
        model.fit(train, attributes=attributes)
        
        train_predictions = model.predict_batch(train)
        test_predictions = model.predict_batch(test)
        
        train_error = calculate_error(train_predictions, train['label'])
        test_error = calculate_error(test_predictions, test['label'])
        
        results['Criterion'].append(criterion)
        results['Max Depth'].append(depth)
        results['Train Error'].append(train_error)
        results['Test Error'].append(test_error)


results_df = pd.DataFrame(results)
print(results_df)


# In[ ]:




