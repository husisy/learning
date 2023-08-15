import numpy as np
import opt_einsum

def demo_opt_einsum_contract_path():
    key='bdik,acaj,ikab,ajac,ikbd'
    tmp0 = set(y for x in key.split(',') for y in x)
    char_to_id = {y:x for x,y in enumerate(tmp0)}
    char_to_dim = {x:np.random.randint(5, 11) for x in char_to_id.keys()}
    shape_list = [[char_to_dim[y] for y in x] for x in key.split(',')]
    array_list = [np.random.randn(*x) for x in shape_list]
    tmp0 = [[char_to_id[y] for y in x] for x in key.split(',')]
    input_sequence = [y for x in zip(array_list,tmp0) for y in x]
    path,path_info = opt_einsum.contract_path(*input_sequence, [])
    result = opt_einsum.contract(*input_sequence, [], optimize=path)


def demo_contract_expression():
    N0 = 3
    N1 = 4
    N2 = 5
    np_rng = np.random.default_rng()
    hf0 = lambda *x: np_rng.uniform(-1, 1, size=x)
    np0 = hf0(N0)
    np1 = hf0(N0, N1)
    np2 = hf0(N0, N2)
    ret_ = opt_einsum.contract(np0, [0], np1, [0,1], np2, [0,2], [1,2])
    contract_expr = opt_einsum.contract_expression((N0,), (0,), (N0,N1), (0,1), (N0,N2), (0,2), (1,2))
    ret0 = contract_expr(np0, np1, np2)
    np.abs(ret_-ret0).max()
