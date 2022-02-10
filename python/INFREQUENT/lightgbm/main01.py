import os
import json
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

# load data
df_train = pd.read_csv(os.path.join('data','regression.train'), header=None, sep='\t')
df_val = pd.read_csv(os.path.join('data','regression.test'), header=None, sep='\t')

X_train = df_train.iloc[:,1:].values #(np,float,(7000,28))
y_train = df_train[0].values
X_val = df_val.iloc[:,1:].values #(np,float,(500,28))
y_val = df_val[0].values

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# parameters
params = {'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0}

# train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_val, early_stopping_rounds=5)
gbm.save_model('model.txt') # save model

y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
print('The rmse of prediction is:', mean_squared_error(y_val, y_pred) ** 0.5)
