import numpy as np
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
print('DT score', cross_val_score(clf, X, y).mean())

clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
print('RF score', cross_val_score(clf, X, y).mean())

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
print('ET score', cross_val_score(clf, X, y).mean())
