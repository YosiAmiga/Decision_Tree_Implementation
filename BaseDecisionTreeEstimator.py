import numpy as np
import Node
import pandas as pd
from Node import Node

"""
Base Decision Tree Class:
The parent of Classification and Regressor Trees.
each field is adjust according to the selected DT type.
"""
class BaseDecisionTreeEstimator:
    def __init__(self,
                 tol,
                 max_depth,
                 min_members,
                 criterion,
                 split_method,
                 max_features):
        self.tol = tol
        self.max_depth = max_depth
        self.min_members = min_members
        self.criterion = criterion
        self.split_method = split_method
        self.max_features = max_features

    # a method to fit the given training data of the features and the predicted classes.
    # X and y are the data to be given to train on.
    def fit(self, X, y, weights=None):
        self.tree_ = Node()
        X_ = self._get_values(X)
        y_ = self._get_values(y)
        self.weights_ = weights
        # decide the split method
        if self.split_method == 'binary':
            feature_types = None
        elif self.split_method == 'nary':
            feature_types = [self.__check_type(X_[:, column]) for column in range(X.shape[1])]
        else:
            raise ValueError('parameter split_method must be binary or nary')
        # create the tree
        self.__generate_tree(self.tree_, X_, y_, weights, feature_types)

    # a method to create the tree from the given X,y data, weights if necessary, and features types
    def __generate_tree(self, tree, X, y, weights, feature_types):
        if len(y) <= self.min_members:
            self._label_node(tree, y)
            return

        if self.tol and tree.calc_impurity(y, store=True, criterion=self.criterion) < self.tol:
            self._label_node(tree, y)
            return

        if self.max_depth and self.tree_.get_depth() >= self.max_depth:
            self._label_node(tree, y)
            return

        best_feature_split = self.__split_attribute(tree, X, y, weights, feature_types)
        tree.feature = best_feature_split[0]
        tree.split = best_feature_split[1]

        if tree.feature is None or tree.split is None:
            self._label_node(tree, y)
            return

        splitted_data = tree.get_split_indices(X)
        num_branches = len(splitted_data)
        if num_branches < 2:
            self._label_node(tree, y)
            return
        elif num_branches == 2:
            if len(splitted_data[0]) == 0 or len(splitted_data[1]) == 0:
                self._label_node(tree, y)
                return

        for branch_indices in splitted_data:
            new_node = Node()
            tree.children.append(new_node)
            branch_weights = weights[branch_indices] if weights is not None else None
            self.__generate_tree(new_node, X[branch_indices], y[branch_indices], branch_weights, feature_types)

    # Method to decide the split attribute from the given X and y data.
    def __split_attribute(self, tree, X, y, weights, feature_types=None):
        min_impurity = np.inf
        impurity = min_impurity
        best_feature = None
        best_split_value = None
        # a list for all the features from given data
        features = []
        #Arange the data from lowest to highest value
        if not self.max_features:
            features = np.arange(X.shape[1])
        elif self.max_features == 'auto':
            features = np.random.choice(X.shape[1], size=np.sqrt(X.shape[1]).astype('int'), replace=False)
        else:
            features = np.random.choice(X.shape[1], size=self.max_features, replace=False)
        #Run on all the features and for each one calculate the impurity.
        #The best one will be the root of the DT.
        #after sorting all the features and doing several calculations, return the:
        # best feature, best split value, and best impurity,
        for feature in features:
            tree.feature = feature
            X_feature = X[:, feature]
            if feature_types is not None and feature_types[feature] == 'cat':
                tree.split = np.unique(X_feature)
                if len(tree.split) < 2:
                    continue
                impurity = tree.impurity_for_split(X, y, weights, criterion=self.criterion)
                if impurity < min_impurity:
                    min_impurity = impurity
                    best_feature = feature
                    best_split_value = tree.split
            else:
                X_feature_sorted_indices = np.argsort(X_feature)
                X_feature_sorted = X_feature[X_feature_sorted_indices]
                y_sorted = y[X_feature_sorted_indices]
                thresholds = (X_feature_sorted[1:] + X_feature_sorted[:-1]) / 2
                thresholds_len = len(thresholds)
                for value_index, value in enumerate(thresholds):
                    if (value_index < thresholds_len - 1) and (
                            y_sorted[value_index] == y_sorted[value_index + 1] or thresholds[value_index] == thresholds[
                        value_index + 1]):
                        continue

                    tree.split = value
                    impurity = tree.impurity_for_split(X, y, weights, criterion=self.criterion)
                    if impurity < min_impurity:
                        min_impurity = impurity
                        best_feature = feature
                        best_split_value = tree.split

        return best_feature, best_split_value, min_impurity

    def _label_node(self, node, y):
        pass
    # The method to decide for a given X data and
    def _decide(self, node, X, pred, indices):
        #If it is a leaf node, set the prediction to be the label of the node and return.
        if node.leaf:
            pred[indices] = node.label
            return
        #Get all the X data values as branches and use the split method.
        #for each branch recursivly call the decide method until reaching a leaf and setting the label.
        branches = node.get_split_indices(X, indices)
        for index, branch in enumerate(branches):
            self._decide(node.children[index], X, pred, branch)
    #for a given data, check its data type
    def __check_type(self, data):
        try:
            number_data = data.astype(np.number)
            if np.all(np.mod(number_data, 1) == 0):
                return 'cat' if len(np.unique(data)) / len(data) <= 0.05 else 'num'
            return 'num'
        except ValueError:
            return 'cat'
    #for a given data, check if it is a known format of a dataframe and return the values.
    def _get_values(self, data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values
        return data