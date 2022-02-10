import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesClassifier

data = fetch_olivetti_faces()
X = data.data #ori(400,64,64)
y = data.target

mask = y<5  # Limit to 5 classes (originally 40 class)
X = X[mask]
y = y[mask]

clf = ExtraTreesClassifier(n_estimators=1000, max_features=128, n_jobs=1, random_state=0)
clf.fit(X, y)
importances = clf.feature_importances_.reshape(data.images[0].shape)
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances with forests of trees")
