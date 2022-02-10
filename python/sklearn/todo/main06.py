import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

iris = datasets.load_iris()
ind1 = np.random.permutation(np.where(iris.target!=0)[0]) #remove label 2
num1 = ind1.shape[0]
ind_train = ind1[:round(num1*0.9)]
ind_val = ind1[round(num1*0.9):]
data_train = iris.data[ind_train,:2]
label_train = iris.target[ind_train]
data_val = iris.data[ind_val,:2]
label_val = iris.target[ind_val]

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(data_train, label_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(data_train[:,0], data_train[:,1], c=label_train, zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(data_val[:,0], data_val[:,1], s=80, facecolors='none', zorder=10, edgecolor='k')

    tmp1 = np.linspace(data_train[:,0].min()-1, data_train[:,0].max()+1)
    tmp2 = np.linspace(data_train[:,1].min()-1, data_train[:,1].max()+1)
    xx,yy = np.meshgrid(tmp1, tmp2)
    zz = clf.decision_function(np.stack([xx.ravel(),yy.ravel()], axis=1)).reshape(xx.shape)

    plt.pcolormesh(xx, yy, zz>0, cmap=plt.cm.Paired)
    plt.contour(xx, yy, zz, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()
