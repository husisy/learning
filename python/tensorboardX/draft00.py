import os
import numpy as np

from tensorboardX import SummaryWriter

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

writer = SummaryWriter(hf_file('exp-0'))
np0 = np.sin(np.linspace(0, 2*np.pi, 100))
for ind0 in range(np0.size):
    writer.add_scalar('sin', np0[ind0], ind0)
