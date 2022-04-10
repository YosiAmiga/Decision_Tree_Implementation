from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

"""Split the data into clf test & train"""

df = pd.read_csv('data.csv')  # Convert csv file to data frame
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


def test_decision_Tree_clf(df):
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

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print("\nBest accuracy OF SKLEARN CLF DECISION TREE: " + str(+clf.score(X_test, y_test)))



def test_decision_Tree_regressor(df):
    df['area_type'] = pd.factorize(df['area_type'])[0]
    # print(df['area_type'] )
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

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    cross_val_score(regressor, X_train, y_train, cv=10)
    print('THIS IS THE SKLEARN REGRESSOR SCORE: ',regressor.score(X_validation,y_validation))


"""SCORE OF DECISION TREE CLASSIFICATION ALGORITHM"""
test_decision_Tree_clf(df=pd.read_csv('data.csv'))
test_decision_Tree_regressor(df=pd.read_csv('data.csv'))
