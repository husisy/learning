import numpy as np
import numpy.linalg
import sklearn.decomposition

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def demo_decomposition_pca():
    # see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    num_sample = 1000
    num_feature = 10
    num_pca_components = 3
    np_data = np.random.randn(num_sample, num_feature)

    # PCA via sklearn
    pca = sklearn.decomposition.PCA(n_components=num_pca_components)
    pca.fit(np_data)
    np2 = pca.transform(np_data)
    sklearn_pca_output = pca.transform(np_data)

    # PCA via numpy
    tmp1 = np_data - np_data.mean(axis=0)
    tmp2 = np.matmul(tmp1.transpose(1,0), tmp1)/(tmp1.shape[0]-1)
    EVL,EVC = np.linalg.eigh(tmp2)
    EVL_kept = EVL[::-1][:num_pca_components]
    EVC = EVC.T[::-1][:num_pca_components]

    # compare sklearn-results with numpy-results
    assert hfe(pca.explained_variance_, EVL_kept) < 1e-5
    assert hfe(pca.explained_variance_ratio_, EVL_kept/EVL.sum()) < 1e-5

    #EVC_i is parallel to pca.componenets_, but no guarentte to be exactly same
    tmp1 = np.abs(np.sum(EVC * pca.components_, axis=1))
    assert np.abs(tmp1-1).max() < 1e-5

    # to be exactly same as sklearn_pca_output, we have to use pca.components_
    np_pca_output = (np_data-np_data.mean(axis=0)) @ pca.components_.T
    assert hfe(sklearn_pca_output, np_pca_output) < 1e-5
