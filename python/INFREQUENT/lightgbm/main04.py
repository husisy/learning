import numpy as np
import lightgbm as lgb

np.random.seed(0)

n_sample = 1000
n_feature = 4
X = np.random.rand(n_sample, n_feature)
tmp1 = (X + 0.2*(np.random.rand(n_sample,n_feature)-0.5)).sum(axis=1)
y = (((tmp1 - tmp1.min())/(tmp1.max()-tmp1.min()))>0.75).astype(np.int32)

ind1 = np.random.permutation(n_sample)
ind_train = ind1[:round(n_sample*0.7)]
ind_val = ind1[round(n_sample*0.7):]
X_train,y_train = X[ind_train],y[ind_train]
X_val,y_val = X[ind_val],y[ind_val]
# w_train = np.random.rand(X_train.shape[0])

ds_train = lgb.Dataset(X_train, label=y_train)
ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)

# param = {'num_leaves':31,
#         'num_trees':1000,
#         'is_unbalance':True,
#         'objective':'binary',
#         'metric':['auc'],
#         'verbose':-1,
# }

# bst = lgb.train(param, ds_train, valid_sets=[ds_train,ds_val], early_stopping_rounds=20)
# print('is_unbalance: {}'.format('is_unbalance' in param))


param = {'num_leaves':31,
        'num_trees':1000,
        # 'is_unbalance':True,
        'objective':'binary',
        'metric':['auc'],
        'lambda_l2':10,
        'verbose':-1,
}

bst = lgb.train(param, ds_train, valid_sets=[ds_val,ds_train], early_stopping_rounds=20)
print('is_unbalance: {}'.format('is_unbalance' in param))
print(np.sum(y_val==0), np.sum(y_val==1))
# bst.save_model('model.txt')

# np1 = np.random.rand(10,3)
# hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
# bst_ = lgb.Booster(model_file='model.txt')
# print('rel error: ', hfe(bst.predict(np1), bst.predict(np1)))
