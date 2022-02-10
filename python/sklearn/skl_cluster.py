import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-3: round(hfe(x,y,eps),5)

def skl_KMeans():
    tmp1 = load_iris()
    np1 = tmp1['data']
    np2 = tmp1['target']

    clf = KMeans(3)
    clf.fit(np1)
    np3 = clf.predict(np1)

    distance = np.sum((np1[:,np.newaxis] - clf.cluster_centers_[np.newaxis])**2, axis=2)
    np3_ = np.argmin(distance, axis=1)
    print('skl_KMeans:: np vs skl: ', np.all(np3==np3_))


if __name__=='__main__':
    skl_KMeans()
