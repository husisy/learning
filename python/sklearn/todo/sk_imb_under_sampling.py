from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
        NearMiss, InstanceHardnessThreshold, CondensedNearestNeighbour, EditedNearestNeighbours,
        RepeatedEditedNearestNeighbours, AllKNN, NeighbourhoodCleaningRule, OneSidedSelection)

def create_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98), n_classes=3, class_sep=0.8, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_classes=n_classes,
            n_clusters_per_class=n_clusters, weights=list(weights), class_sep=class_sep, random_state=0)

def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_sample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)

def plot_decision_function(X, y, clf, ax):
    tmp1 = np.linspace(X[:,0].min()-1, X[:,0].max()+1)
    tmp2 = np.linspace(X[:,1].min()-1, X[:,1].max()+1)
    xx, yy = np.meshgrid(tmp1, tmp2)
    zz = clf.predict(np.stack([xx.ravel(),yy.ravel()], axis=1)).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')

# ClusterCentroids
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94), class_sep=0.8)
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
sampler = ClusterCentroids(random_state=0)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_decision_function(X, y, clf, ax2)
ax2.set_title('Decision function for {}'.format(sampler.__class__.__name__))
plot_resampling(X, y, sampler, ax3)
ax3.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()

# RandomUnderSampler
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94), class_sep=0.8)

clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
sampler = RandomUnderSampler(random_state=0)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_decision_function(X, y, clf, ax2)
ax2.set_title('Decision function for {}'.format(sampler.__class__.__name__))
plot_resampling(X, y, sampler, ax3)
ax3.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()

# NearMiss
fig, ((ax1,ax3,ax5), (ax2,ax4,ax6)) = plt.subplots(2, 3, figsize=(12, 9))
# X, y = create_dataset(n_samples=5000, weights=(0.1, 0.2, 0.7), class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6))
tmp1 = (NearMiss(version=1,random_state=0), NearMiss(version=2,random_state=0), NearMiss(version=3,random_state=0))
for ax, sampler in zip(ax_arr, tmp1):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}-{}'.format(sampler.__class__.__name__, sampler.version))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}-{}'.format(sampler.__class__.__name__, sampler.version))
fig.tight_layout()

# EditedNearestNeighbours
fig, ((ax1,ax3,ax5), (ax2,ax4,ax6)) = plt.subplots(2, 3, figsize=(12, 9))
# X, y = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6))
tmp1 = (EditedNearestNeighbours(random_state=0),
        RepeatedEditedNearestNeighbours(random_state=0),
        AllKNN(random_state=0, allow_minority=True))
for ax, sampler in zip(ax_arr, tmp1):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}'.format(sampler.__class__.__name__))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()

# 
fig, ((ax1,ax3,ax5), (ax2,ax4,ax6)) = plt.subplots(2, 3, figsize=(12, 9))
# X, y = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)

ax_arr = ((ax1, ax2), (ax3, ax4), (ax5, ax6))
tmp1 = (CondensedNearestNeighbour(random_state=0),
        OneSidedSelection(random_state=0),
        NeighbourhoodCleaningRule(random_state=0))
for ax, sampler in zip(ax_arr, tmp1):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0])
    ax[0].set_title('Decision function for {}'.format(sampler.__class__.__name__))
    plot_resampling(X, y, sampler, ax[1])
    ax[1].set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()

#
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94), class_sep=0.8)

clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
sampler = InstanceHardnessThreshold(random_state=0, estimator=LogisticRegression())
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_decision_function(X, y, clf, ax2)
ax2.set_title('Decision function for {}'.format(sampler.__class__.__name__))
plot_resampling(X, y, sampler, ax3)
ax3.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()

# plt.show()
