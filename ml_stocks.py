import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def process_data(ticker):
    window_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, window_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02

    for col in cols:
        if col > requirement:
            return 1
        if col < requirement:
            return -1
    return 0


def extract_feature_sets(ticker):
    tickers, df = process_data(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
                                        df['{}_1d'.format(ticker)],
                                        df['{}_2d'.format(ticker)],
                                        df['{}_3d'.format(ticker)],
                                        df['{}_4d'.format(ticker)],
                                        df['{}_5d'.format(ticker)],
                                        df['{}_6d'.format(ticker)],
                                        df['{}_7d'.format(ticker)]
                                        ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print(Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[tick for tick in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def ml_logic(ticker):
    X, y, df = extract_feature_sets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())
                            ])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy ', confidence)
    predictions = clf.predict(X_test)
    print(Counter(predictions))

    return confidence


ml_logic('BAC')