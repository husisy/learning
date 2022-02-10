import os
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# load data
df_train = pd.read_csv(os.path.join('data','regression.train'), header=None, sep='\t')
df_val = pd.read_csv(os.path.join('data','regression.test'), header=None, sep='\t')

X_train = df_train.iloc[:,1:].values #(np,float,(7000,28))
y_train = df_train[0].values
X_val = df_val.iloc[:,1:].values #(np,float,(500,28))
y_val = df_val[0].values

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# train
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train, eval_set=[(X_val,y_val)], eval_metric='l1', early_stopping_rounds=5)
print('Feature importances:', list(gbm.feature_importances_))

y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration_)
print('The rmse of prediction is:', mean_squared_error(y_val, y_pred)**0.5)

# grid search
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {'learning_rate':[0.01,0.1,1], 'n_estimators':[20,40]}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
