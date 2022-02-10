import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

hf1 = lambda x: x*np.sin(x)
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = hf1(x)

# model
plt.figure()
colors = ['teal', 'yellowgreen', 'gold']
plt.plot(x_plot, hf1(x_plot), color='cornflowerblue', linewidth=2, label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(x[:,np.newaxis], y)
    y_plot = model.predict(x_plot[:,np.newaxis])
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=2, label="degree %d"%degree)

plt.legend(loc='lower left')
plt.show()
