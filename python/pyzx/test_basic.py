import numpy as np

import pyzx

def test_zspidar00():
    np_rng = np.random.default_rng()
    phase = np_rng.uniform(0, 1)
    g = pyzx.Graph()
    v0 = g.add_vertex(pyzx.VertexType.BOUNDARY, phase=0)
    v1 = g.add_vertex(pyzx.VertexType.BOUNDARY, phase=0)
    v2 = g.add_vertex(pyzx.VertexType.Z, phase=phase) #in unit of pi
    g.add_edges([(0,2), (1,2)], pyzx.EdgeType.SIMPLE)
    g.set_inputs((0,1))
    # pyzx.draw(g)
    ret0 = g.to_tensor()
    ret_ = np.array([[1,0],[0,np.exp(1j*phase*np.pi)]])
    assert np.abs(ret0-ret_).max() < 1e-10
