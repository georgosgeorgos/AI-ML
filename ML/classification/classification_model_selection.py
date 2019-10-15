import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def routine(clf, parameters):

    grid = GridSearchCV(clf, parameters, scoring="accuracy")
    grid.fit(x_train, y_train)

    clf = grid.best_estimator_
    clf.fit(x_train, y_train)
    best_score = clf.score(x_train, y_train)
    test_score = cross_val_score(clf, x_test, y_test, cv=5, scoring="accuracy").mean()

    return best_score, test_score


with open("input.csv", "r") as f:

    ff = csv.reader(f)
    data = []
    for row in ff:
        data.append(row)

data = np.array(data[1:]).astype(float)

X = data[:, :-1]
Y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1, stratify=Y)

svm_linear = {"C": [0.1, 0.5, 1, 5, 10, 50, 100]}
svm_polynomial = {"C": [0.1, 1, 3], "degree": [4, 5, 6], "gamma": [0.1, 1]}
svm_rbf = {"C": [0.1, 0.5, 1, 5, 10, 50, 100], "gamma": [0.1, 0.5, 1, 3, 6, 10]}
logistic = {"C": [0.1, 0.5, 1, 5, 10, 50, 100]}
knn = {"n_neighbors": [3, 5, 10, 20, 50], "leaf_size": [5, 10, 15, 60]}
decision_tree = {"max_depth": [1, 2, 3, 50], "min_samples_split": [2, 3, 10]}
random_forest = {"max_depth": [1, 2, 3, 50], "min_samples_split": [2, 3, 10]}

classifiers = {
    "svm_linear": svm.SVC(kernel="linear"),
    "svm_polynomial": svm.SVC(kernel="poly"),
    "svm_rbf": svm.SVC(kernel="rbf"),
    "logistic": LogisticRegression(),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
}

parameters = {
    "svm_linear": svm_linear,
    "svm_polynomial": svm_polynomial,
    "svm_rbf": svm_rbf,
    "logistic": logistic,
    "knn": knn,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
}


res = []

for classifier in ["svm_linear", "svm_polynomial", "svm_rbf", "logistic", "knn", "decision_tree", "random_forest"]:

    clf = classifiers[classifier]
    par = parameters[classifier]

    best_score, test_score = routine(clf, par)
    res.append([classifier, best_score, test_score])


with open("output.csv", "w", newline="") as fp:

    file = csv.writer(fp, delimiter=",")
    file.writerows(res)
