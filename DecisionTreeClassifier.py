from BaseDecisionTreeEstimator import BaseDecisionTreeEstimator
import numpy as np
import scipy.stats as st  # use for finding the most common element in an array

"""
Decision Tree Classifier Class:
all the values the a Classifier DT uses are in the constructor of the Base DT that he inherit.
set the default criterion to be Entropy, and the split method to nary.
"""


class DecisionTreeClassifier(BaseDecisionTreeEstimator):
    def __init__(self,
                 tol=None,
                 max_depth=None,
                 min_members=10,
                 criterion='entropy',
                 split_method='nary',
                 max_features=None, sample_weight=None):
        super().__init__(tol, max_depth, min_members, criterion, split_method, max_features)
        self.sample_weight = sample_weight

    # This function calculate the most frequent label node for a given node and label.
    def _label_node(self, node, y):
        most_frequent = st.mode(y)[0]
        rand = np.random.randint(len(most_frequent))
        node.leaf = True
        node.label = most_frequent[rand]

    # The prediction function:
    # For a given data X, predict the class it will be classified to by the created DT Classifier.
    # the function traverse the tree(depending on the feature spliting at each node and its value in X)
    # the class to be classified in will be found in the leaves of the DT.
    def predict(self, X):
        X_ = self._get_values(X)
        pred = np.full(X_.shape[0], -1)
        self._decide(self.tree_, X_, pred, np.arange(X_.shape[0]))
        return pred

    # Returns the score of the given prediction, by using X and y data and comparing it to the actual resault.
    def score(self, X, y):
        y_pred = self.predict(X)
        return y_pred[y == y_pred].size / y_pred.size

    # Print function to visualize the DT Classifier in a printing way
    def __str__(self):
        return "DecisionTreeClassifier(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(
            self.tol, self.max_depth, self.min_members, self.criterion, self.split_method, self.max_features)

    def __repr__(self):
        return "DecisionTreeClassifier(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(
            self.tol, self.max_depth, self.min_members, self.criterion, self.split_method, self.max_features)
