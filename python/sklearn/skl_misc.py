import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

from utils import next_tbd_dir

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-3: round(hfe(x,y,eps),5)

def sklearn_save_model00(model, path):
    with open(path, 'wb') as fid:
        pickle.dump(model, fid)
def sklearn_load_model00(path):
    with open(path, 'rb') as fid:
        return pickle.load(fid)
def sklearn_save_model01(model, path):
    joblib.dump(model, path)
def sklearn_load_model01(path):
    return joblib.load(path)

def test_save_load_model(N0=10, N1=4, N2=3):
    tmp1 = next_tbd_dir()
    hf_file = lambda *x,_dir=tmp1: os.path.join(_dir, *x)
    path = hf_file('test01.pkl') #suffix name doesn't matter

    np1 = np.random.rand(N0,N1)
    np2 = np.random.randint(0, N2, size=[N0,])
    clf = SVC(probability=True)
    clf.fit(np1,np2)
    np3 = clf.predict_proba(np1)

    sklearn_save_model00(clf, path)
    np4 = sklearn_load_model00(path).predict_proba(np1)
    print('sklearn_save_model00: ', hfe_r5(np3,np4))

    sklearn_save_model00(clf, path)
    np4 = sklearn_load_model01(path).predict_proba(np1)
    print('sklearn_save_model01: ', hfe_r5(np3,np4))

if __name__=='__main__':
    test_save_load_model()

