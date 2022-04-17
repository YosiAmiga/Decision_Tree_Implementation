import numpy as np

"""
Node class will represent a feature in the data frame, each node we build in a DT will be from this class
"""
class Node:
    def __init__(self, feature=-1, split=None, impurity=np.inf, n_outputs=None):
        # Split feature
        self.feature = feature
        # Split criterion
        self.split = split
        self.impurity = impurity
        self.children = []
        self.leaf = False
        self.label = None
        self.depth = 0
        self.n_outputs = None

    # Get the depth of the given node, recursive function to calculate the tree height
    def get_depth(self):
        if self.leaf or len(self.children) == 0:
            return 1
        return 1 + max([child.get_depth() for child in self.children])

    def get_split_indices(self, X, intersect_with=None):
        X_feature = X[:, self.feature]

        # Categorical feature
        if isinstance(self.split, np.ndarray):
            splitted_data = []
            for value in self.split:
                indices = np.asarray(X_feature == value).nonzero()[0]
                if intersect_with is not None:
                    indices = np.intersect1d(indices, intersect_with, assume_unique=True)
                splitted_data.append(indices)

            return splitted_data

        # Numerical feature
        indices_left = np.asarray(X_feature < self.split).nonzero()[0]
        indices_right = np.asarray(X_feature >= self.split).nonzero()[0]

        if intersect_with is not None:
            indices_left = np.intersect1d(indices_left, intersect_with, assume_unique=True)
            indices_right = np.intersect1d(indices_right, intersect_with, assume_unique=True)

        return [indices_left, indices_right]

    def __get_probs(self, y):
        unique_y = np.unique(y)
        probs = np.zeros(len(unique_y))
        y_len = len(y)
        for i, y_i in enumerate(unique_y):
            probs[i] = len(y[y == y_i]) / y_len

        return probs
    # Function to calculate the impurity of each feature, we select the criterion of impurity to calculate on.
    def calc_impurity(self, y, criterion, store=False):
        impurity = None

        if criterion == 'entropy':
            probs = self.__get_probs(y)
            impurity = self.__calc_entropy(probs)
        elif criterion == 'gini':
            probs = self.__get_probs(y)
            impurity = self.__calc_gini(probs)
        elif criterion == 'mse':
            impurity = self.__mean_squared_err(y)
        elif criterion == 'mae':
            impurity = self.__mean_absolute_err(y)

        if store:
            self.impurity = impurity
        return impurity
    # use the given criterion to calculate impurity for splitting the date
    def impurity_for_split(self, X, y, weights, criterion):
        splitted_indices = self.get_split_indices(X)
        impurities = np.zeros(len(splitted_indices))
        if weights is not None:
            total_weight = np.sum(weights)
        else:
            total_weight = len(y)
        for index, branch_indices in enumerate(splitted_indices):
            y_branch = y[branch_indices]
            if weights is not None:
                total_branch_weight = np.sum(weights[branch_indices])
            elif criterion == 'mse' or criterion == 'mae':
                total_branch_weight = 1
            else:
                total_branch_weight = len(y_branch)
            impurities[index] = self.calc_impurity(y_branch, criterion) * total_branch_weight
        return np.sum(impurities) / total_weight

    # Functions for each criterion: Entropy, GINI, MSE, MAE.
    def __calc_entropy(self, probs):
        entropy = -np.sum(probs * np.log(probs + 10e-10))
        return entropy

    def __calc_gini(self, probs):
        gini = 1 - np.sum(probs ** 2)
        return gini

    def __mean_squared_err(self, y):
        y_m = np.mean(y)
        return np.sum((y - y_m) ** 2)

    def __mean_absolute_err(self, y):
        y_m = np.median(y)
        return np.abs(y - y_m)

    def __str__(self):
        return 'Node(leaf={})'.format(self.leaf)

    def __repr__(self):
        return 'Node(leaf={})'.format(self.leaf)
    # Print function to visualize the data in a printing way
    def __print_state__(self):
        print(
            'split: \n', self.split, 'impurity ', self.impurity, 'children ', self.children, 'leaf ', self.leaf,
            'label',
            self.label, 'depth ', self.depth)
        return
