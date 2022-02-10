import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
data = iris.data[:,:2] #(150,4)
label = iris.target

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    clf = KNeighborsClassifier(15, weights=weights)
    clf.fit(data, label)

    tmp1 = np.linspace(data[:,0].min()-1, data[:,0].max()+1,100)
    tmp2 = np.linspace(data[:,1].min()-1, data[:,1].max()+1,100)
    xx,yy = np.meshgrid(tmp1, tmp2)
    zz = clf.predict(np.stack([xx.ravel(),yy.ravel()], axis=1)).reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, zz, cmap=cmap_light)

    plt.scatter(data[:,0], data[:,1], c=label, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (15, weights))
