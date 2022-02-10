import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, decomposition, datasets

if __name__ == '__main__':
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target

    logistic = linear_model.LogisticRegression()

    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    # Prediction
    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs), n_jobs=1)
    tmp1 = time.time()
    estimator.fit(X_digits, y_digits)
    print('time elapsed: ', time.time()-tmp1)
