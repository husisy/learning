import numpy as np
from collections import Counter

import projectq
from projectq import MainEngine
from projectq.ops import H, Measure, CNOT, All
from projectq.backends import CircuitDrawer, Simulator

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def projectq_QRNG():
    engine = MainEngine()
    q0 = engine.allocate_qubit()
    H | q0
    Measure | q0
    engine.flush()
    ret = int(q0)
    return ret

# bad warning
# def test_projectq_QRNG():
#     print(Counter(projectq_QRNG() for _ in range(100)))


def misc00():
    engine = MainEngine()
    q0 = engine.allocate_qubit()
    q1 = engine.allocate_qubit()
    H | q0
    CNOT | (q0, q1)
    engine.flush()
    All(Measure) | (q0+q1)
    # CircuitDrawer().get_latex() #fail to compile both in win10 and sharelatex


def test_remove_projectq_warning():
    engine = MainEngine()
    q0 = engine.allocate_qureg(3)
    All(H) | q0
    All(Measure) | q0


def misc01():
    engine = MainEngine(backend=Simulator())
    q0 = engine.allocate_qubit()
    q1 = engine.allocate_qubit()
    H | q0
    z0 = engine.backend.get_amplitude('00', q0+q1)
    engine.flush()
    z1 = engine.backend.get_amplitude('00', q0+q1)
    # engine.backend.get_probability('00',q0+q1)
    # engine.backend.get_expectation_value()
    # engine.backend.set_wavefunction()
    All(Measure) | (q0+q1)

