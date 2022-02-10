import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def plot_result(x,y,pred):
    plt.plot(x,y,'o')
    plt.plot(x,pred,'r')
    plt.show()

n_train = 100
X_train = np.linspace(0, 4*np.pi, n_train)
y_train = np.cos(X_train) + 0.5*(np.random.rand(n_train)-0.5)

X_val = np.linspace(-2*np.pi, 6*np.pi, 200)

regr1 = GradientBoostingRegressor(learning_rate=0.1, n_estimators=20, max_depth=5)
regr1.fit(X_train[:,np.newaxis], y_train)
pred1 = regr1.predict(X_val[:,np.newaxis])

regr2 = RandomForestRegressor(n_estimators=20)
regr2.fit(X_train[:,np.newaxis], y_train)
pred2 = regr2.predict(X_val[:,np.newaxis])

line1 = plt.plot(X_train, y_train, 'rx', label='train data')[0]
line2 = plt.plot(X_val, pred1, label='GBDT')[0]
line3 = plt.plot(X_val, pred2, label='random forest')[0]
plt.legend(handles=[line1, line2, line3])

plt.gcf().show()
