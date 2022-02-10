import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33)

clf = svm.SVC(gamma=0.001)
clf.fit(x_train, y_train)
prediction_test = clf.predict(x_test)

print(metrics.classification_report(y_test, prediction_test))
print(metrics.confusion_matrix(y_test, prediction_test))
