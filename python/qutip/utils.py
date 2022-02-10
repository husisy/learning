import numpy as np

import qutip
from qutip import Qobj, ket2dm

hf_randn_c = lambda *size: np.random.randn(*size) + np.random.randn(*size)*1j


def generate_random_ket(N0):
    return Qobj(hf_randn_c(N0, 1)).unit()


def generate_random_density_matrix(N0):
    tmp0 = np.random.rand(N0)
    probability = tmp0/tmp0.sum()
    ret = 0
    for probability_i in probability:
        ret = ret + probability_i*ket2dm(Qobj(hf_randn_c(N0, 1)).unit())
    return ret


def generate_random_operator(N0, is_hermite=False):
    if is_hermite:
        return generate_random_density_matrix(N0)
    else:
        return generate_random_density_matrix(N0) + 1j*generate_random_density_matrix(N0)
