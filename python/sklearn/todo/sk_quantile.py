import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=2)

iris = load_iris()
X,y = iris.data, iris.target
qt_transformer = QuantileTransformer()
X_trans = qt_transformer.fit_transform(X)

print(np.percentile(X[:,0], [0,25,50,75,100]))
print(np.percentile(X_trans[:,0], [0,25,50,75,100]))
print(type(qt_transformer.quantiles_), qt_transformer.quantiles_.shape)
