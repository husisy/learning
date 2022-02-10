import os
import numpy as np
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.externals import joblib

boston = load_boston()
X = boston.data
y = boston.target
hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
np1 = np.random.rand(1,X.shape[1])

# sklearn
select_k_best = SelectKBest(f_regression, k=round(X.shape[1]/2))
svr = SVR(kernel='linear')
pipe = Pipeline(steps=[('select_k_best',select_k_best), ('svr',svr)])
pipe.fit(X,y)

# implement with numpy
mask = pipe.named_steps['select_k_best'].get_support()
svr_coef = pipe.named_steps['svr'].coef_
svr_intercept = pipe.named_steps['svr'].intercept_

x1 = pipe.predict(np1)
x2 = np.dot(np1[mask[np.newaxis]], svr_coef.transpose((1,0))) + svr_intercept
print('rel error: ', hfe(x1,x2))

# save and load
tmp1 = os.path.join('tbd01','tbd01.pkl')
joblib.dump(pipe, tmp1)
pipe_ = joblib.load(tmp1)
x3 = pipe_.predict(np1)
print('rel error: ', hfe(x1,x3))
