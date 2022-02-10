import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize, Bounds, LinearConstraint, curve_fit
from lightgbm import LGBMClassifier
from mlens.visualization import corrmat

np.set_printoptions(precision=3)

# load data
# digits = load_digits()
# X = digits.data
# y = digits.target
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int32)
num1 = X.shape[0]
ind1 = np.random.permutation(num1)
ind_train = ind1[:round(num1*0.5)]
ind_val = ind1[round(num1*0.5):round(num1*0.6)]
ind_test = ind1[round(num1*0.6):]
X_train,y_train = X[ind_train], y[ind_train]
X_val,y_val = X[ind_val], y[ind_val]
X_test,y_test = X[ind_test], y[ind_test]

# base estimator
n_est = 5
tmp1 = np.linspace(2, 15, n_est).astype(np.int32)
print(tmp1)
clf_list = [RandomForestClassifier(n_estimators=10, max_features=x) for x in tmp1]
# clf_list.append(LGBMClassifier(boosting_type='gbdt', num_leaves=31, objective='multiclass', n_estimators=3))
# clf_list.append(LGBMClassifier(boosting_type='gbdt', num_leaves=31, objective='multiclass', n_estimators=10))
n_est = len(clf_list)
for ind1 in range(n_est):
    print(ind1)
    clf_list[ind1].fit(X_train, y_train)

# super learner
tmp1 = np.concatenate([clf.predict_proba(X_val) for clf in clf_list], axis=1)
super_learner = LogisticRegression()
super_learner.fit(tmp1, y_val)


hf1 = lambda label,pred: np.mean(label==np.argmax(pred, axis=1))
base_pred = [clf.predict_proba(X_test) for clf in clf_list]
for ind1 in range(n_est):
    print('base estimator_{} acc: {:.3f}'.format(ind1, hf1(y_test, base_pred[ind1])))
print('naive average acc: {:.3f}'.format(hf1(y_test, sum(base_pred)/n_est)))

tmp1 = stats.mode(np.stack([np.argmax(x,axis=1) for x in base_pred], axis=1), axis=1)[0][:,0]
print('majority voting acc: {:.3f}'.format(np.mean(y_test==tmp1)))

tmp1 = super_learner.predict_proba(np.concatenate(base_pred, axis=1))
print('super learner acc: {:.3f}'.format(hf1(y_test, tmp1)))
