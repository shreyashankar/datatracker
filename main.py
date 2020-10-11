import numpy as np
import pandas as pd
import time

from datatracker import DataTracker
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

random_state = 0

X, y = load_breast_cancer(return_X_y=True)
dt = DataTracker(X)

clf = LogisticRegression(random_state=random_state).fit(X, y)

def predict(X, clf):
    return clf.predict_proba(X)[:, 1]

def confidence(X, clf):
    return clf.decision_function(X)

dt.add_tracker('prediction', predict)
dt.add_tracker('confidence', confidence)

dt.track(X=X, clf=clf)

dt.dump_results()