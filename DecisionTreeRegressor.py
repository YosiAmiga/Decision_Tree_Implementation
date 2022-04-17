from BaseDecisionTreeEstimator import BaseDecisionTreeEstimator
import numpy as np

"""
Decision Tree Regressor Class:
all the values the a Classifier DT uses are in the constructor of the Base DT that he inherit.
set the default criterion to be MSE, and the split method to nary.
"""
class DecisionTreeRegressor(BaseDecisionTreeEstimator):
    def __init__(self,
                 tol=None,
                 max_depth=None,
                 min_members=10,
                 criterion='mse',
                 split_method='nary',
                 max_features=None):
        super().__init__(tol, max_depth, min_members, criterion, split_method, max_features)
    # This function calculate the mean/median depending on the criterion.
    # In out case it is always mean, default criterion is MSE.
    def _label_node(self, node, y):
        node.leaf = True
        if self.criterion == 'mse':
            node.label = np.mean(y)
        elif self.criterion == 'mae':
            node.label = np.median(y)
    #The prediction function:
    #For a given data X, predict the price (number) it will have to by the created DT Regressor.
    def predict(self, X):
        X_ = self._get_values(X)
        pred = np.full(X_.shape[0], 0.0)
        # print('this is the prediction vector:', pred)
        self._decide(self.tree_, X_, pred, np.arange(X_.shape[0]))
        # print('this is pred after deision tree traversal ', pred)
        return pred

    # Returns the score of the given prediction, by using X and y data and comparing it to the actual result.
    def score(self, X, y):
        y_pred = self.predict(X) # vector of predictions --> numerically (regressor)
        y_m = np.mean(y) # calculate the mean(AVG) of the vector

        u= ((y - y_pred)** 2).sum()
        v=((y - y.mean()) ** 2).sum()

        """where u is the residual sum of squares ((y_true - y_pred)** 2).sum()"""

        """ v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()"""
        ss_reg = np.sum((y_pred - y_m) ** 2) # sum of squared r
        ss_tot = np.sum((y - y_m) ** 2)
        score= 1-(u/v)
        return score

    def __str__(self):
        return "DecisionTreeRegressor(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(
            self.tol, self.max_depth, self.min_members, self.criterion, self.criterion, self.split_method)

    def __repr__(self):
        return "DecisionTreeRegressor(tol={}, max_depth={}, min_members={}, criterion={}, split_method={}, max_features={})".format(
            self.tol, self.max_depth, self.min_members, self.criterion, self.criterion, self.split_method)