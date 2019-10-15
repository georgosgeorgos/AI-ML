import os
import re
import glob
import copy
import nltk
import sklearn
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import linear_model
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


def process(file, stop):

    file = file.lower()
    file = re.sub("<[^<]+>", "", file)
    file = re.sub("[^\w-]", " ", file)
    file = file.split(" ")
    file = [i.strip() for i in file if i.strip() not in stop]
    # file = [i for i in file if i not in stop]
    string = " ".join(file)

    return string


def preprocessTrainTest(PATH_TRAIN_POS, PATH_TRAIN_NEG, PATH_STOP, PATH_TEST):

    with open(PATH_STOP) as fp:
        stop = fp.read()
        stop = set(stop.lower().split("\n"))
        stop = stop.union(set(stopwords.words("english")))

    pos = os.listdir(PATH_TRAIN_POS)
    neg = os.listdir(PATH_TRAIN_NEG)
    sentiment = []

    for j in range(len(pos)):
        file = None
        with open(PATH_TRAIN_POS + pos[j]) as fp:
            file = fp.read()

        string = process(file, stop)
        score = pos[j].split("_")[1].split(".")[0]

        if int(score) > 5:
            label = 1
        else:
            label = 0

        row = [string, label]

        sentiment.append(row)

    for j in range(len(neg)):
        file = None

        with open(PATH_TRAIN_NEG + neg[j]) as fp:
            file = fp.read()

        string = process(file, stop)
        score = neg[j].split("_")[1].split(".")[0]

        if int(score) > 5:
            label = 1
        else:
            label = 0

        row = [string, label]

        sentiment.append(row)

    sentimentDataFrame = pd.DataFrame(data=sentiment, columns=["text", "polarity"])
    for i in range(100):
        sentimentDataFrame = sentimentDataFrame.sample(frac=1, replace=False, random_state=np.random.RandomState())

    sentimentDataFrame.to_csv("imdb_tr.csv", columns=["text", "polarity"])
    sentiment = sentimentDataFrame.values

    test_0 = pd.read_csv(PATH_TEST, encoding="latin1")
    test_0 = list(test_0["text"].values)

    test = []
    for j in range(len(test_0)):

        string = process(test_0[j], stop)
        row = [j, string]
        test.append(row)

    return sentiment, test


def selectModel(x, y, test_size):

    parameters = {"alpha": [10 ** i for i in range(-10, 1, 1)]}

    sgd = linear_model.SGDClassifier(penalty="l1", loss="hinge", n_iter=100, n_jobs=-1, shuffle=True)

    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=test_size)
    scaler = MaxAbsScaler()  # attention!!! Same scaling for train and test
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_cv = scaler.transform(x_cv)

    grid = GridSearchCV(sgd, parameters, scoring="accuracy")
    grid.fit(x_train, y_train)
    best_score = grid.score(x_train, y_train)
    test_score = cross_val_score(grid, x_cv, y_cv, cv=5, scoring="accuracy").mean()

    return grid, test_score  # probably cross_validation doesn't work with sparse matrices


def predictor(sentiment, test, n_gram, tf_idf, test_size):

    X = 0
    Y = 0
    X_test = 0

    if tf_idf == True:
        if n_gram == True:
            ext_tdidf = feature_extraction.text.TfidfVectorizer(
                ngram_range=(1, 2), stop_words="english", max_features=5000
            )
        else:
            ext_tdidf = feature_extraction.text.TfidfVectorizer(
                ngram_range=(1, 1), stop_words="english", max_features=5000
            )

        X = ext_tdidf.fit_transform([sent[0] for sent in sentiment])
        Y = np.array([sent[1] for sent in sentiment])

    else:
        if n_gram == True:
            ext = feature_extraction.text.CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
        else:
            ext = feature_extraction.text.CountVectorizer(ngram_range=(1, 1), stop_words="english", max_features=5000)

        X = ext.fit_transform([sent[0] for sent in sentiment])
        Y = np.array([sent[1] for sent in sentiment])

    if tf_idf == True:
        X_test = ext_tdidf.transform([t[1] for t in test])
    else:
        X_test = ext.transform([t[1] for t in test])

    # clf_opt, test_score = selectModel(X,Y,test_size)

    sgd = linear_model.SGDClassifier(
        penalty="l1", loss="hinge", n_iter=100, n_jobs=-1, shuffle=True, alpha=0.0001
    )  # clf_opt.best_params_["alpha"])

    scaler = MaxAbsScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)

    sgd.fit(X, Y)
    print(sgd.score(X, Y))
    prediction = sgd.predict(X_test)

    return prediction
