import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

X, y = make_classification(n_samples=5000, n_features=2, n_classes=3, n_redundant=0,
        n_clusters_per_class=1, weights=[0.01, 0.05, 0.94], class_sep=0.8, random_state=0)

print('before oversampling: ', Counter(y))

over_sampler = RandomOverSampler()
# over_sampler = SMOTE()
# over_sampler = ADASYN()
X_resampled, y_resampled = over_sampler.fit_sample(X, y)
print('after oversampling: ', Counter(y_resampled))
print('after oversampling[:5000]', Counter(y_resampled[:5000]))

X_resampled, y_resampled = SMOTE().fit_sample(X, y)
