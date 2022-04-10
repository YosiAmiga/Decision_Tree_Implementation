from BaseDecisionTreeEstimator import BaseDecisionTreeEstimator
import numpy as np
import scipy.stats as st


class DecisionTreeClassifier(BaseDecisionTreeEstimator):
    def __init__(self,
                 tol=None,
                 max_depth=None,
                 min_members=10,
                 criterion='entropy',
                 split_method='nary',
                 max_features=None,sample_weight=None):
        super().__init__(tol, max_depth, min_members, criterion, split_method, max_features)
        self.sample_weight=sample_weight

    def _label_node(self, node, y):
        most_frequent = st.mode(y)[0]
        rand = np.random.randint(len(most_frequent))
        node.leaf = True
        node.label = most_frequent[rand]

    def predict(self, X):
        X_ = self._get_values(X)
        pred = np.full(X_.shape[0], -1)
        self._decide(self.tree_, X_, pred, np.arange(X_.shape[0]))
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return y_pred[y == y_pred].size / y_pred.size

    def __str__(self):
        return "DecisionTreeClassifier(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(
            self.tol, self.max_depth, self.min_members, self.criterion, self.split_method, self.max_features)

    def __repr__(self):
        return "DecisionTreeClassifier(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(
            self.tol, self.max_depth, self.min_members, self.criterion, self.split_method, self.max_features)
