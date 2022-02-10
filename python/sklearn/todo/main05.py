import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


iris = datasets.load_iris()
data = iris.data[:,:2]
label = iris.target

models = (svm.SVC(kernel='linear', C=1),
          svm.LinearSVC(C=1),
          svm.SVC(kernel='rbf', gamma=0.7, C=1),
          svm.SVC(kernel='poly', degree=3, C=1))
models = (clf.fit(data, label) for clf in models)

titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

tmp1 = np.linspace(data[:,0].min()-1, data[:,0].max()+1)
tmp2 = np.linspace(data[:,1].min()-1, data[:,1].max()+1)
xx,yy = np.meshgrid(tmp1,tmp2)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(data[:,0], data[:,1], c=label, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()
