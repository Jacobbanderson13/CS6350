import numpy as np
import pandas as pd
import math
from collections import Counter

class ID3:
    def __init__(self, max_depth=None, criterion='information_gain'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = {}
        self.train_df = None
    
    def fit(self, df, target_attribute_name="y", attributes=None, depth=0):
        if depth == 0:
            self.train_df = df.copy()
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
        best_attr = None
        best_gain = -float('inf')

        for attr in attributes:
            if self.criterion == 'information_gain':
                gain = self.information_gain(df, attr, target_attribute_name)
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

    def information_gain(self, df, attribute, target_attribute_name):
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