import numpy as np
import stim
import beliefmatching
import itertools
import numft

np_rng = np.random.default_rng()

def test_random_SpF2_matrix():
    for n in [1,2,3,4]:
        tmp0 = stim.Tableau.random(n).to_numpy() #6-tuple
        x0 = np.stack([tmp0[0],tmp0[1],tmp0[2],tmp0[3]]).reshape(2,2,n,n).transpose(0,2,1,3).reshape(2*n,2*n).astype(np.uint8)
        x1 = np.kron(np.array([[0,1],[1,0]]), np.eye(n)).astype(np.uint8)
        assert np.all((x0 @ x1 @ x0.T) % 2 == x1)

    x0 = stim.Tableau.random(3)
    tmp0 = x0.to_numpy()
    x1 = stim.Tableau.from_numpy(x2x=tmp0[0], x2z=tmp0[1], z2x=tmp0[2], z2z=tmp0[3], x_signs=tmp0[4], z_signs=tmp0[5])
    assert x0==x1


def test_PauliString():
    x0 = stim.PauliString('-iIXYZ')
    assert x0.sign==-1j
    xbit,zbit = x0.to_numpy()
    assert np.all(xbit==np.array([0,1,1,0], dtype=np.uint8))
    assert np.all(zbit==np.array([0,0,1,1], dtype=np.uint8))
    x1 = stim.PauliString.from_numpy(xs=xbit, zs=zbit, sign=-1j)
    assert x0==x1

    x0 = stim.PauliString('XIX')
    x1 = stim.PauliString('IXZ')
    assert x0*x1 == stim.PauliString('-iXXY')


def test_Tableau_CNOT():
    x0 = stim.Circuit('CX 0 1').to_tableau() #stim.Tableau
    assert x0.x_output(0)==stim.PauliString('XX')
    assert x0.z_output(0)==stim.PauliString('ZI')
    assert x0.x_output(1)==stim.PauliString('IX')
    assert x0.z_output(1)==stim.PauliString('ZZ')


def test_tableau_inverse():
    for n in [1,2,3,4]:
        x0 = stim.Tableau.random(n)
        assert (x0 * x0.inverse()) == stim.Tableau(n)
        assert (x0.inverse() * x0) == stim.Tableau(n)


def test_tableau_bell():
    # bell circuit
    x0 = stim.Circuit('H 0\nI 1').to_tableau()
    x1 = stim.Circuit('CX 0 1').to_tableau()
    x2 = x0.then(x1)
    assert x2.z_output(0)==stim.PauliString('XX')
    assert x2.z_output(1)==stim.PauliString('ZZ')


def test_classical_control():
    circ = stim.Circuit(
    '''
    H 0
    M 0
    CX rec[-1] 1
    M 0 1
    DETECTOR rec[-2] rec[-3]
    DETECTOR rec[-1] rec[-2]
    ''')
    x0 = circ.compile_sampler()
    tmp0 = x0.sample(10).astype(np.uint8)
    assert np.all(tmp0[:,0]==tmp0[:,1]) and np.all(tmp0[:,1]==tmp0[:,2])
    x1 = circ.compile_detector_sampler()
    assert np.all(x1.sample(10)==0)


def get_stim_check_matrix(circ:stim.Circuit)->dict[str,np.ndarray]:
    tag_set = set()
    circ1 = stim.Circuit()
    for inst in circ:
        if inst.name=="DEPOLARIZE1":
            for q in inst.targets_copy():
                tag_set.add(inst.tag)
                p = inst.gate_args_copy()[0]
                circ1.append('X_ERROR', q.value, arg=p, tag=f'{inst.tag}_X')
                circ1.append('Z_ERROR', q.value, arg=p, tag=f'{inst.tag}_Z')
        else:
            circ1.append(inst)
    dem = circ1.detector_error_model()
    x0 = beliefmatching.detector_error_model_to_check_matrices(dem)
    check_new = x0.check_matrix.toarray().T.copy()
    tmp0 = (x.tag for x in dem if x.type=='error')
    tmp1 = {x:i for i,x in enumerate(tmp0)}
    z0 = {k:check_new[[tmp1[k+'_X'], tmp1[k+'_Z']]] for k in tag_set}
    return z0


def test_c513_stim_circuit_noise():
    c513_stab = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
    c513_logicalZ = 'ZZZZZ'
    c513_logicalX = 'XXXXX'
    c513_logicalY = 'YYYYY'
    tmp0 = [stim.PauliString(x) for x in c513_stab]
    tmp1 = [stim.PauliString(x) for x in [c513_logicalZ, c513_logicalX, c513_logicalY]]
    tmp0.append(tmp1[np_rng.integers(0,3)] * np_rng.choice([-1,1])) #either logical is ok for the following test
    x0 = stim.Tableau.from_stabilizers(tmp0)
    circ_prep = x0.to_circuit()
    tmp0 = ''
    for i0,x0 in enumerate(c513_stab, start=circ_prep.num_qubits):
        tmp0 = tmp0 + f'\nH {i0}'
        tmp0 = tmp0 + '\n' + '\n'.join([f'C{y} {i0} {i1}' for i1,y in enumerate(x0) if y!='I'])
        tmp0 = tmp0 + f'\nH {i0}' + f'\nM {i0}'
        tmp0 = tmp0 + f'\nDETECTOR[stab{i0-circ_prep.num_qubits}] rec[-1]'
    circ_syndrome = stim.Circuit(tmp0)
    physical_error_rate = 0.1
    circ_error = stim.Circuit('\n'.join([f'DEPOLARIZE1[err{i}]({physical_error_rate}) {i}' for i in range(circ_prep.num_qubits)]))
    circ = circ_prep + circ_error + circ_syndrome
    assert circ.count_determined_measurements()==len(c513_stab)
    x0 = circ.detector_error_model(decompose_errors=True)
    x1 = beliefmatching.detector_error_model_to_check_matrices(x0)
    # x1.check_matrix.toarray()
    '''[[1 1 1 1 1 1 1 0 0 0 1 0 0 0 0]
        [0 1 1 0 0 0 1 1 1 1 1 0 0 1 0]
        [0 0 1 1 1 0 1 0 1 0 0 1 1 1 0]
        [0 0 0 0 1 1 1 0 0 1 1 0 1 1 1]]'''
    assert set(((1<<np.arange(len(c513_stab))) @ x1.check_matrix.toarray()).tolist())==set(range(1,2**len(c513_stab)))
    assert np.abs(1-(1-physical_error_rate)**(1/3) - x1.priors).max() < 1e-4

    check_dict = get_stim_check_matrix(circ)
    check = np.stack([check_dict[f'err{x}'] for x in range(5)]).transpose(2,1,0).reshape(4,10)

    tmp0 = check.reshape(4,2,5)[:,::-1].reshape(4,10)
    assert np.all(numft.pauli.str_to_GF4(c513_stab)==tmp0)

    sampler = circ.compile_detector_sampler() #stim.CompiledDetectorSampler
    syndrome = sampler.sample(10) #shape=(10,4)

    measurement_sampler = circ.compile_sampler() #stim.CompiledMeasurementSampler
    measurements = measurement_sampler.sample(10) #shape=(10,4)
    # m2d = circ.compile_m2d_converter()

    dem = circ.detector_error_model(decompose_errors=True)
    sampler = dem.compile_sampler() #stim.CompiledDemSampler
    syndrome,_,ebit = sampler.sample(10, return_errors=True)
