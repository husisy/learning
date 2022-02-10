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
