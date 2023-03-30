import numpy as np


np_rng = np.random.default_rng()
np0 = np_rng.standard_normal(23)
np0 = np_rng.random(23)
np1 = np_rng.integers(0, 233, size=(23,))

np_rng = np.random.default_rng(seed=233)
np_rng = np.random.Generator(np.random.PCG64(seed=233))
np_rng = np.random.Generator(np.random.MT19937(seed=233))
# TODO parallel generation: https://numpy.org/doc/stable/reference/random/index.html#parallel-generation
