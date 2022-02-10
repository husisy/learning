import numpy as np
from sklearn.feature_selection import VarianceThreshold

x = np.array([[0,0,1], [0,1,0], [1,0,0], [0,1,1], [0,1,0], [0,1,1]])
print('var: ', x.var(axis=0))
sel = VarianceThreshold(threshold=0.2)
sel.fit_transform(x) #delete first column