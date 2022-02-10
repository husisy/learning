import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

def hf_softmax(data, axis=-1):
    tmp0 = data - data.max(axis=axis, keepdims=True)
    tmp1 = np.exp(tmp0)
    return tmp1 / tmp1.sum(axis=axis, keepdims=True)
hf_logistic = lambda x: 1 / (1 + np.exp(-x))

# import some data to play with
iris = datasets.load_iris()
data = iris.data[:,:2]
label = iris.target

clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='ovr')
clf.fit(data, label)

tmp1 = np.linspace(data[:,0].min()-1, data[:,0].max()+1)
tmp2 = np.linspace(data[:,1].min()-1, data[:,1].max()+1)
xx,yy = np.meshgrid(tmp1, tmp2)
zz = clf.predict(np.stack([xx.ravel(),yy.ravel()], axis=1)).reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, zz, cmap=plt.cm.Paired)

plt.scatter(data[:,0], data[:, 1], c=label, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
