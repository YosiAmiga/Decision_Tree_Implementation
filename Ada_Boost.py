import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import DecisionTreeClassifier

""" HELPER FUNCTION: GET ERROR RATE ========================================="""


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""


def print_error_rate(err):
    err = 1-err
    'Error rate: Training: %.4f - Test: %.4f' % err


""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""


def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    # print('pred test ', pred_test)
    # print('pred train ', pred_train)
    print('The Adaboost score we implemented is', 1-get_error_rate(pred_test, Y_test))
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


""" ADABOOST IMPLEMENTATION ================================================="""


def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights , fit means for us building a stump
        clf.fit(X_train, Y_train, weights=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x == 1 else -1 for x in miss]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


""" PLOT FUNCTION ==========================================================="""


def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticks(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')
    plt.show()
    plt.pause(100)


""" MAIN SCRIPT ============================================================="""


def main():
    df = pd.read_csv('data.csv')  # Convert csv file to data frame

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
    Y_train = y_classes.iloc[:8041, ]
    y_validation = y_classes.iloc[8041:10051, ]
    Y_test = y_classes.iloc[10051:12563, ]

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier.DecisionTreeClassifier()
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    stumps = 50
    er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, stumps, clf_tree)
    # er_i = adaboost_clf(y_validation, X_validation, Y_test, X_test, stumps, clf_tree)
    er_train.append(er_i[0])
    er_test.append(er_i[1])

