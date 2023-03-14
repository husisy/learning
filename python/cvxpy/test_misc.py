import numpy as np
import cvxpy

def get_example_data(N0,N1,N2):
    np_rng = np.random.default_rng()
    tmp0 = np_rng.normal(size=(N0*N1, 2*N0*N1))
    tmp1 = tmp0 @ tmp0.T
    np0 = tmp1 / np.trace(tmp1)
    tmp0 = np0.reshape(N0,N1,N0,N1).transpose(0,2,1,3).reshape(N0*N0, N1*N1)
    np1 = np_rng.normal(size=(N1*N1,N2))
    np2 = tmp0 @ np1
    return np0,np1,np2

def test_cvxpy_tensor_transpose_fortran_order():
    # also see scipy/test_core.py/test_reshape_c_fortran_order()
    N0 = 3
    N1 = 5
    N2 = 7

    np0,np1,np2 = get_example_data(N0,N1,N2)
    index_fortran = np.arange(N0*N1*N0*N1).reshape(N0,N1,N0,N1).transpose(3,1,2,0).reshape(-1)

    cvxX = cvxpy.Variable((N0*N1,N0*N1))
    cvx_tmp0 = cvxpy.reshape(cvxX, N0*N1*N0*N1, order='F')[index_fortran]
    cvx_tmp1 = cvxpy.reshape(cvx_tmp0, (N0*N0,N1*N1), order='F')
    constraints = [
        cvxX>>0,
        cvxX==cvxX.T,
        cvxpy.trace(cvxX)==1,
        cvx_tmp1 @ np.asfortranarray(np1) == np.asfortranarray(np2),
    ]
    obj = cvxpy.Minimize(1)
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()
    print('prob:', prob.value)
    assert prob.value is not None

    solution = np.ascontiguousarray(cvxX.value)
    assert np.abs(solution-solution.T).max() < 1e-7 #symmetric matrix
    print('min(eig(np0))', np.linalg.eigvalsh(solution).min()) #semi-definite positive
    print('trace(np0)', np.trace(np0)) #should be 1
    assert np.abs(solution.reshape(N0,N1,N0,N1).transpose(0,2,1,3).reshape(N0*N0,N1*N1) @ np1 - np2).max() < 1e-10
