import numpy as np


rng = np.random.default_rng()
np0 = rng.standard_normal(23)
np0 = rng.random(23)
np1 = rng.integers(0, 233, size=(23,))

rng = np.random.default_rng(seed=233)
rng = np.random.Generator(np.random.PCG64(seed=233))
rng = np.random.Generator(np.random.MT19937(seed=233))
# TODO parallel generation: https://numpy.org/doc/stable/reference/random/index.html#parallel-generation
