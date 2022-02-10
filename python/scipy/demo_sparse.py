import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def demo_lil_construct_matrix():
    snp0 = scipy.sparse.lil_matrix((1000,1000))
    snp0[0,:100] = np.random.rand(100)
    snp0[1,100:200] = snp0[0,:100]
    snp0.setdiag(np.random.rand(1000))
    snp0_csr = snp0.tocsr()


def demo_csr_matrix_vector_multiply_eficiency():
    N0 = 1024
    snp0 = scipy.sparse.rand(N0, N0, density=0.01, format='csr')
    np1 = np.random.randn(N0)
    np0 = snp0.toarray()
    # %timeit ret_ = np.dot(np0, np1) #443 µs ± 2.46 µs #ipython
    # %timeit ret0 = snp0.dot(np1) #10.1 µs ± 37.7 ns #ipython


def demo_sparse_random():
    N0 = 500

    z0 = scipy.sparse.rand(N0, N0, density=0.1, format='csr')
    ind0,ind1 = z0.nonzero()
    # value = z0.tolil()[ind0,ind1].toarray()[0]
    value = z0.data
    print('scipy.sparse.rand(): #nonzero:', value.size) #25000
    print('scipy.sparse.rand(): min:', value.min()) #>=0
    print('scipy.sparse.rand(): max:', value.max()) #<=1
    print('scipy.sparse.rand(): mean:', value.mean()) #~0.5
    print('scipy.sparse.rand(): std:', value.std(ddof=1)) #~0.289

    mean = 0.233
    std = 2.33
    hf_rand = lambda x: np.random.normal(mean, std, size=x)
    z1 = scipy.sparse.random(N0, N0, density=0.1, data_rvs=hf_rand, format='csr')
    ind0,ind1 = z1.nonzero()
    value = z1.tolil()[ind0,ind1].toarray()[0]
    print('scipy.sparse.random(): #nonzero: ', value.size)
    print('scipy.sparse.random(): mean: ', value.mean()) #~0.233
    print('scipy.sparse.random(): std: ', value.std(ddof=1)) #~2.33
