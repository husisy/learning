import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='GPU')


def test_bad_default_initialize():
    # bad behavior, set seed once, new parameter once
    ms.common.seed.set_seed(23)
    z0 = np.stack([ms.nn.Dense(2,3).weight.asnumpy().reshape(-1) for _ in range(6)])
    assert np.all(z0.std(axis=0)<1e-5)
    ms.common.seed.set_seed(233)
    z1 = np.stack([ms.nn.Dense(2,3).weight.asnumpy().reshape(-1) for _ in range(6)])
    assert np.all(z1.std(axis=0)<1e-5)
    assert np.abs(z0-z1).mean() > 1e-4

    # acceptable behavior
    ms.common.seed.set_seed(233)
    init = ms.common.initializer.HeNormal()
    z0 = np.stack([ms.nn.Dense(2,3,weight_init=init).weight.asnumpy().reshape(-1) for _ in range(6)])
    ms.common.seed.set_seed(233)
    init = ms.common.initializer.HeNormal()
    z1 = np.stack([ms.nn.Dense(2,3,weight_init=init).weight.asnumpy().reshape(-1) for _ in range(6)])
    assert np.abs(z0[0] - z0[1]).mean() > 1e-4
    assert np.abs(z1[0] - z1[1]).mean() > 1e-4
    assert np.abs(z0[0] - z1[0]).max() < 1e-5
    assert np.abs(z0[1] - z1[1]).max() < 1e-5

def test_bad_seed_behavior():
    # bad behavior, change the seed of numpy
    ms.common.seed.set_seed(233)
    np0 = np.random.rand(2,3)
    ms.common.seed.set_seed(233)
    np1 = np.random.rand(2,3)
    assert np.abs(np0-np1).max() < 1e-5

    # walkaround
    ms.common.seed.set_seed(233)
    rng = np.random.default_rng()
    np0 = rng.random((2,3))
    ms.common.seed.set_seed(233)
    rng = np.random.default_rng()
    np1 = rng.random((2,3))
    assert np.abs(np0-np1).mean() > 1e-4
