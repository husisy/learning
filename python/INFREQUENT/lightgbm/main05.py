import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_digits

np.random.seed(0)

digits = load_digits()
X = digits.data
X = X + np.random.rand(*X.shape)*10
y = digits.target
n_sample = X.shape[0]

ind1 = np.random.permutation(n_sample)
ind_train = ind1[:round(n_sample*0.5)]
ind_val = ind1[round(n_sample*0.5):]
X_train,y_train = X[ind_train],y[ind_train]
X_val,y_val = X[ind_val],y[ind_val]

ds_train = lgb.Dataset(X_train, label=y_train)
ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

param = {'num_leaves':30,
        'num_trees':1000,
        'objective':'binary',
        'metric':['auc'],
        'verbose':-1,
        'early_stopping_round':20,
}
print('first run')
bst = lgb.train(param, ds_train, valid_sets=[ds_val], verbose_eval=10)
print('\n second run')
bst = lgb.train(param, ds_train, valid_sets=[ds_val], verbose_eval=10)

param = {'num_leaves':30,
        'num_trees':1000,
        'objective':'binary',
        'metric':['auc'],
        'verbose':-1,
}
print('first run after putting early_stopping_rounds in lgb.train()')
bst = lgb.train(param, ds_train, valid_sets=[ds_val], early_stopping_rounds=20, verbose_eval=10)
print('\n second run')
bst = lgb.train(param, ds_train, valid_sets=[ds_val], early_stopping_rounds=20, verbose_eval=10)