import numpy as np
import cupy as cp
import cuquantum

import numpysim

hfe = lambda x, y, eps=1e-3: np.max(np.abs(x - y) / (np.abs(x) + np.abs(y) + eps))

def cuq_apply_single_gate(q0, op, index, handle, num_qubit):
    nTargets = 1
    nControls = 0
    adjoint = 0
    index = num_qubit - 1 - index
    extraWorkspaceSizeInBytes = cuquantum.custatevec.apply_matrix_get_workspace_size(
        handle, cuquantum.cudaDataType.CUDA_C_64F, num_qubit, op.data.ptr, cuquantum.cudaDataType.CUDA_C_64F,
        cuquantum.custatevec.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_64F)
    if extraWorkspaceSizeInBytes > 0:
        workspace_ptr = cp.cuda.alloc(extraWorkspaceSizeInBytes).ptr
    else:
        workspace_ptr = 0
    controls = ()
    cuquantum.custatevec.apply_matrix(
        handle, q0.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, num_qubit,
        op.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, cuquantum.custatevec.MatrixLayout.ROW, adjoint,
        (index,), nTargets, controls, 0, len(controls), cuquantum.ComputeType.COMPUTE_64F,
        workspace_ptr, extraWorkspaceSizeInBytes)


def cuq_apply_double_gate(q0, op, index0, index1, handle, num_qubit):
    nTargets = 2
    nControls = 0
    adjoint = 0
    assert index0!=index1
    index0 = num_qubit - 1 - index0
    index1 = num_qubit - 1 - index1
    index0,index1 = index1,index0
    extraWorkspaceSizeInBytes = cuquantum.custatevec.apply_matrix_get_workspace_size(
        handle, cuquantum.cudaDataType.CUDA_C_64F, num_qubit, op.data.ptr, cuquantum.cudaDataType.CUDA_C_64F,
        cuquantum.custatevec.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_64F)
    if extraWorkspaceSizeInBytes > 0:
        workspace_ptr = cp.cuda.alloc(extraWorkspaceSizeInBytes).ptr
    else:
        workspace_ptr = 0
    controls = ()
    cuquantum.custatevec.apply_matrix(
        handle, q0.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, num_qubit,
        op.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, cuquantum.custatevec.MatrixLayout.ROW, adjoint,
        (index0,index1), nTargets, controls, 0, len(controls), cuquantum.ComputeType.COMPUTE_64F,
        workspace_ptr, extraWorkspaceSizeInBytes)


def cuq_single_expectation(q0, op, index, handle, num_qubit):
    nTargets = 1
    index = num_qubit - 1 - index
    extraWorkspaceSizeInBytes = cuquantum.custatevec.compute_expectation_get_workspace_size(handle,
        cuquantum.cudaDataType.CUDA_C_64F, num_qubit,
        op.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, cuquantum.custatevec.MatrixLayout.ROW,
        nTargets, cuquantum.ComputeType.COMPUTE_64F)
    if extraWorkspaceSizeInBytes > 0:
        workspace_ptr = cp.cuda.alloc(extraWorkspaceSizeInBytes).ptr
    else:
        workspace_ptr = 0
    ret = np.zeros(1, dtype=np.complex128)
    cuquantum.custatevec.compute_expectation(handle, q0.data.ptr, cuquantum.cudaDataType.CUDA_C_64F,
        num_qubit, ret.__array_interface__['data'][0], cuquantum.cudaDataType.CUDA_C_64F,
        op.data.ptr, cuquantum.cudaDataType.CUDA_C_64F, cuquantum.custatevec.MatrixLayout.ROW,
        (index,), nTargets, cuquantum.ComputeType.COMPUTE_64F, workspace_ptr, extraWorkspaceSizeInBytes)
    return ret.item()


def rand_state(num_qubit):
    tmp0 = (np.random.randn(2**num_qubit) + 1j*np.random.randn(2**num_qubit)).astype(np.complex128)
    ret = tmp0 / np.linalg.norm(tmp0)
    return ret

def rand_unitary_matrix(N0, tag_complex=True, seed=None):
    np_rng = np.random.default_rng(seed)
    if tag_complex:
        tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T.conj()
    else:
        tmp0 = np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T
    ret = np.linalg.eigh(tmp0)[1]
    return ret


def test_apply_single_operator(num_qubit=5):
    ind0 = int(np.random.randint(num_qubit))
    q0 = rand_state(num_qubit)
    operator = rand_unitary_matrix(2, tag_complex=True)
    ret_ = numpysim.np_apply_gate(q0, operator, [ind0])

    h_cuq0 = cuquantum.custatevec.create()
    q0_cp = cp.array(q0, dtype=cp.complex128)
    op_cp = cp.array(operator, dtype=cp.complex128).reshape(-1)
    cuq_apply_single_gate(q0_cp, op_cp, ind0, h_cuq0, num_qubit)
    cuquantum.custatevec.destroy(h_cuq0)
    ret0 = q0_cp.get()
    assert hfe(ret_, ret0) < 1e-5


def test_apply_double_operator(num_qubit=5):
    ind0,ind1 = np.random.permutation(num_qubit)[:2].tolist()
    q0 = rand_state(num_qubit)
    operator = rand_unitary_matrix(4, tag_complex=True)
    ret_ = numpysim.np_apply_gate(q0, operator, [ind0,ind1])

    h_cuq0 = cuquantum.custatevec.create()
    q0_cp = cp.array(q0, dtype=cp.complex128)
    op_cp = cp.array(operator, dtype=cp.complex128).reshape(-1)
    cuq_apply_double_gate(q0_cp, op_cp, ind0, ind1, h_cuq0, num_qubit)
    cuquantum.custatevec.destroy(h_cuq0)
    ret0 = q0_cp.get()
    assert hfe(ret_, ret0) < 1e-5


def test_single_operator_expectation(num_qubit=5):
    q0 = rand_state(num_qubit)
    operator = np.random.randn(2,2) + 1j*np.random.randn(2,2)
    ind0 = np.random.randint(num_qubit)
    ret_ = numpysim.np_operator_expectation(q0, operator, [ind0])

    h_cuq0 = cuquantum.custatevec.create()
    q0_cp = cp.array(q0, dtype=cp.complex128)
    op_cp = cp.array(operator, dtype=cp.complex128).reshape(-1)
    ret0 = cuq_single_expectation(q0_cp, op_cp, ind0, h_cuq0, num_qubit)
    cuquantum.custatevec.destroy(h_cuq0)
    assert abs(ret_-ret0)<1e-6
