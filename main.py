import pandas as pd
from time import time
from functools import reduce
from DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTreeRegressor import DecisionTreeRegressor
import Ada_Boost


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
def DT_implementation_CLF_setUP(df):
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

    X_train = X_Features.iloc[:8041, :]
    X_validation = X_Features.iloc[8041:10051, :]
    X_test = X_Features.iloc[10051:12563, :]

    y_train = y_classes.iloc[:8041, ]
    y_validation = y_classes.iloc[8041:10051, ]
    y_test = y_classes.iloc[10051:12563, ]

    params_to_optimize = {
        'tol': [0.1],
        'max_depth': [6],
        'min_members': [10, 20, 50],
        'criterion': ['gini'],
        'split_method': ['binary'],
        'max_features': [2, 7]
    }

    best_dt_accuracy, best_dt_model = grid_search(DecisionTreeClassifier, params_to_optimize, X_train, y_train, X_test,
                                                  y_test)
    best_dt_model.fit(X_train, y_train)
    print("DECISION TREE CLF ACCURACY SCORE: " + str(+best_dt_accuracy))

    """TEST SAMPLE optional for visualization"""
    # test_sample = X_train.iloc[33:34, :]
    # res_prediction = best_dt_model.predict(test_sample)
    # print('\n depth of the clasification tree: ', best_dt_model.tree_.get_depth())

# Implementation of the reggresion decision tree
def DT_implementation_REG_setUP(df):
    ######################## REGRESSION DECISION TREE ########################

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

    reg_features = list(df.columns[1:8])
    X_reg = df[reg_features]
    y_reg = df['price in rupees']

    X_train = X_reg.iloc[:8041, :]
    X_validation = X_reg.iloc[8041:10051, :]
    X_test = X_reg.iloc[10051:12563, :]

    # y_@ - split the classes data by rows that were requested, from the y_classes column only!
    y_train = y_reg.iloc[:8041, ]
    y_validation = y_reg.iloc[8041:10051, ]
    y_test = y_reg.iloc[10051:12563, ]

    reg_tree = DecisionTreeRegressor()
    reg_tree.fit(X_train, y_train)
    print('DECISION TREE REG ACCURACY SCORE: ', reg_tree.score(X_test, y_test))


DT_implementation_CLF_setUP(df=pd.read_csv('data.csv'))
DT_implementation_REG_setUP(df=pd.read_csv('data.csv'))

Ada_Boost.main()
