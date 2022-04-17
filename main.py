import pandas as pd
from time import time
from functools import reduce
from DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTreeRegressor import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import Ada_Boost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
# model evaluation
from sklearn.metrics import accuracy_score, mean_squared_error

"""
Helper functions
"""

# a method to split the
def train_test_split(X, ratio=0.8):
    X = X.sample(frac=1).reset_index(drop=True)
    return X[:int(len(X) * ratio)], X[int(len(X) * ratio):]


# Helper functions for grid search
def cartesian_product(arr1, arr2):
    product = []
    for i in arr1:
        for j in arr2:
            if isinstance(i, tuple):
                # (1, 2) and 5 -> (1, 2, 5)
                product.append((*i, j))
            else:
                # 1 and 2 -> (1, 2)
                product.append((i, j))
    return product


def all_possible_param_combinations(params):
    return reduce(cartesian_product, map(lambda param_name: params[param_name], params))


def grid_search(model, params_to_optimize, X_train, y_train, X_test, y_test):
    all_possibilities = all_possible_param_combinations(params_to_optimize)
    best_accuracy = 0
    best_model = None
    for index, possibility in enumerate(all_possibilities):
        model_i = model(*possibility)
        a = time()
        model_i.fit(X_train, y_train)
        b = time()
        # print('model', index + 1)
        # print('trained in', b - a, 'seconds')
        accuracy = model_i.score(X_test, y_test)
        #         print('accuracy: ', accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_i
        return best_accuracy, best_model

    return best_accuracy, best_model

# Implementation of the classifier decision tree
def Classifier_DT_Creation_And_Testing(df):
    ######################## CLF DECISION TREE ########################
    """Split the data into clf test & train"""
    df['area_type'] = pd.factorize(df['area_type'])[0]
    # Prepare the data data
    features = list(df.columns[2:9])
    # y - the target classification to determine for each given data
    # x - all the features in the tree to help determine the final classification for each data
    y_classes = df['area_type']
    X_Features = df[features]

    # split the data into: train , validation , test sub dataframes
    # X_@ - split the features data by rows that were requested,  from the X_Features column only!
    X_train = X_Features.iloc[:8041, :]
    X_validation = X_Features.iloc[8041:10051, :]
    X_test = X_Features.iloc[10051:12563, :]

    # y_@ - split the classes data by rows that were requested, from the y_classes column only!
    y_train = y_classes.iloc[:8041, ]
    y_validation = y_classes.iloc[8041:10051, ]
    y_test = y_classes.iloc[10051:12563, ]

    # the hyper parameters to build the decision tree on
    params_to_optimize = {
        'tol': [0.1],
        'max_depth': [6],
        'min_members': [10, 20, 50],
        'criterion': ['gini'],
        'split_method': ['binary'],
        'max_features': [2, 7]
    }
    # the best_dt_model is the tree
    best_dt_accuracy, best_dt_model = grid_search(DecisionTreeClassifier, params_to_optimize, X_train, y_train, X_test,
                                                  y_test)
    # call fit method to build the tree
    best_dt_model.fit(X_train, y_train)
    print("The Classifier DT score we implemented is: " + str(+best_dt_accuracy))

    """TEST SAMPLE optional for visualization"""
    # test_sample = X_train.iloc[33:34, :]
    # res_prediction = best_dt_model.predict(test_sample)
    # print('\n depth of the clasification tree: ', best_dt_model.tree_.get_depth())

# Implementation of the regression decision tree
def Regression_DT_Creation_And_Testing(df):
    ######################## REGRESSION DECISION TREE ########################
    df['area_type'] = pd.factorize(df['area_type'])[0]
    # the hyper parameters to build the decision tree on
    params_to_optimize = {
        'n_learners': [50, 100, 150, 200],
        'n_iters_stop': [5],
        'loss_tol': [10e-4],
        'alpha': [0.1, 0.3, 0.5, 0.8],
        'tol': [0.1],
        'max_depth': [3],
        'min_members': [10, 20],
        'split_method': ['binary'],
        'max_features': [7],
    }
    # Prepare the data data
    # y - the target feature to estimate a number(price) for.
    # x - all the features in the tree to help determine the final estimation for each data.
    reg_features = list(df.columns[1:8])
    X_reg = df[reg_features]
    y_reg = df['price in rupees']

    # X_@ - split the features data by rows that were requested,  from the X_Features column only!
    X_train = X_reg.iloc[:8041, :]
    X_validation = X_reg.iloc[8041:10051, :]
    X_test = X_reg.iloc[10051:12563, :]

    # y_@ - split the classes data by rows that were requested, from the y_classes column only!
    y_train = y_reg.iloc[:8041, ]
    y_validation = y_reg.iloc[8041:10051, ]
    y_test = y_reg.iloc[10051:12563, ]
    reg_tree = DecisionTreeRegressor()
    reg_tree.fit(X_train, y_train)
    print('The Regression DT score we implemented is: ', reg_tree.score(X_test, y_test))

# Testing both the DT implementation
Classifier_DT_Creation_And_Testing(df=pd.read_csv('data.csv'))
Regression_DT_Creation_And_Testing(df=pd.read_csv('data.csv'))

################ COMPARE TO SKLEARN ALGORITHMS ###################
"""Split the data into clf test & train"""

df = pd.read_csv('data.csv')  # Convert csv file to data frame

def test_decision_Tree_clf(df):
    ######################## CLF DECISION TREE ########################
    """Split the data into clf test & train"""
    df['area_type'] = pd.factorize(df['area_type'])[0]
    # Prepare the data data
    features = list(df.columns[2:9])
    # y - the target classification to determine for each given data
    # x - all the features in the tree to help determine the final classification for each data
    y_classes = df['area_type']
    X_Features = df[features]

    # split the data into: train , validation , test sub dataframes
    # X_@ - split the features data by rows that were requested,  from the X_Features column only!
    X_train = X_Features.iloc[:8041, :]
    X_validation = X_Features.iloc[8041:10051, :]
    X_test = X_Features.iloc[10051:12563, :]

    # y_@ - split the classes data by rows that were requested, from the y_classes column only!
    y_train = y_classes.iloc[:8041, ]
    y_validation = y_classes.iloc[8041:10051, ]
    y_test = y_classes.iloc[10051:12563, ]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print("SKLEARN CLF SCORE: " + str(+clf.score(X_test, y_test)))

def test_decision_Tree_regressor(df):
    # Prepare the data data
    # y - the target feature to estimate a number(price) for.
    # x - all the features in the tree to help determine the final estimation for each data.
    df['area_type'] = pd.factorize(df['area_type'])[0]
    reg_features = list(df.columns[1:8])
    X_reg = df[reg_features]
    y_reg = df['price in rupees']

    # X_@ - split the features data by rows that were requested,  from the X_Features column only!
    X_train = X_reg.iloc[:8041, :]
    X_validation = X_reg.iloc[8041:10051, :]
    X_test = X_reg.iloc[10051:12563, :]

    # y_@ - split the classes data by rows that were requested, from the y_classes column only!
    y_train = y_reg.iloc[:8041, ]
    y_validation = y_reg.iloc[8041:10051, ]
    y_test = y_reg.iloc[10051:12563, ]

    dt = DecisionTreeRegressor(random_state=10)

    # define parameter grid
    parameters_grid = {
        'max_depth': [2, 3],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [2, 8]
    }
    grid_search = GridSearchCV(estimator=dt, param_grid=parameters_grid, cv=10)
    grid_search.fit(X_train, y_train)
    print('SKLEARN REGRESSOR SCORE:',grid_search.score(X_test,y_test))

def test_adaboost_from_sklearn(df):

    Y_train = df.iloc[1:8040]
    Y_train.drop(
        ['availability', 'Unnamed: 0', 'bedrooms', 'total_sqft', 'bath', 'balcony', 'ranked', 'price in rupees'],
        axis=1, inplace=True)

    X_train = df.iloc[1:8040]
    X_train.drop(['area_type', 'Unnamed: 0'], axis=1, inplace=True)

    Y_vald = df.iloc[8041:10051]
    Y_vald.drop(
        ['availability', 'Unnamed: 0', 'bedrooms', 'total_sqft', 'bath', 'balcony', 'ranked', 'price in rupees'],
        axis=1, inplace=True)

    X_vald = df.iloc[8041:10051]
    X_vald.drop(['area_type', 'Unnamed: 0'], axis=1, inplace=True)

    Y_test = df.iloc[10051:12563]
    Y_test.drop(
        ['availability', 'Unnamed: 0', 'bedrooms', 'total_sqft', 'bath', 'balcony', 'ranked', 'price in rupees'],
        axis=1, inplace=True)

    X_test = df.iloc[10051:12563, 2:].values

    # define parameter grid
    parameters_grid = {
        'n_estimators': [20, 50]
    }

    ab = AdaBoostClassifier(random_state=0)
    # define grid search
    grid_search = GridSearchCV(estimator=ab, param_grid=parameters_grid, cv=10)
    grid_search.fit(X_train, Y_train)
    best = grid_search.best_estimator_
    # predict
    y_pred = best.predict(X_test)
    # calculate accuracy
    acc = round(accuracy_score(Y_test, y_pred), 3)

    print('Adaboost Score of Sklearn is ',acc)

def skl_reg():
    df = pd.read_csv('data.csv')
    df = pd.get_dummies(df, columns=['area_type'], drop_first=True)
    df = df.astype({"balcony": "int", "availability": "int", "bedrooms": "int", "total_sqft": "int", "bath": "int",
                    "price in rupees": "int"})

    Y_train = df.iloc[1:8040]
    Y_train.drop(['availability', 'Unnamed: 0', 'bedrooms', 'total_sqft', 'bath', 'balcony', 'ranked', 'area_type_P'],
                 axis=1, inplace=True)

    X_train = df.iloc[1:8040]
    X_train.drop(['price in rupees', 'Unnamed: 0'], axis=1, inplace=True)

    Y_vald = df.iloc[8041:10051]
    Y_vald.drop(['availability', 'Unnamed: 0', 'bedrooms', 'total_sqft', 'bath', 'balcony', 'ranked', 'area_type_P'],
                axis=1, inplace=True)

    X_vald = df.iloc[8041:10051]
    X_vald.drop(['price in rupees', 'Unnamed: 0'], axis=1, inplace=True)

    Y_test = df.iloc[10051:12563]
    Y_test.drop(['availability', 'Unnamed: 0', 'bedrooms', 'total_sqft', 'bath', 'balcony', 'ranked', 'area_type_P'],
                axis=1, inplace=True)

    X_test = df.iloc[10051:12563]
    X_test.drop(['price in rupees', 'Unnamed: 0'], axis=1, inplace=True)
    X_test = df.iloc[10051:12563, 1:-1].values
    dt = DecisionTreeRegressor(random_state=10)

    # define parameter grid
    parameters_grid = {
        'max_depth': [2, 3],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [2, 8]
    }
    grid_search = GridSearchCV(estimator=dt, param_grid=parameters_grid, cv=10)
    # fit estimator
    grid_search.fit(X_train, Y_train)

    # get best estimator
    best = grid_search.best_estimator_

    # predict
    y_pred = best.predict(X_test)
    print(dt.score(X_test,Y_test))
    # calculate MSE
    MSE = round(mean_squared_error(Y_test, y_pred), 1)


df=pd.read_csv('data.csv')
test_decision_Tree_clf(df)
test_decision_Tree_regressor(df)
test_adaboost_from_sklearn(df)
Ada_Boost.main()
# Testing the Ada boost implementation


