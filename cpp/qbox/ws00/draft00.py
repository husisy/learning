import os
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt
plt.ion()

hf_file = lambda *x: os.path.join(*x)

def parse_qbox_output(filepath):
    with open(filepath, 'rb') as fid:
        z0 = etree.fromstring(fid.read())
    return z0

z0 = parse_qbox_output(hf_file('ws00', 'output'))
np0 = np.array([float(x) for x in z0.xpath('//etotal_int/text()')])
fig,ax = plt.subplots()
ax.plot(np0)
ax.grid()


z0 = parse_qbox_output(hf_file('ws00', 'cg.output'))
np0 = np.array([float(x) for x in z0.xpath('//etotal/text()')])
fig,ax = plt.subplots()
ax.plot(np0)
ax.grid()


z0 = parse_qbox_output(hf_file('ws00', 'md.output'))
np0 = np.array([float(x) for x in z0.xpath('//etotal/text()')])
np1 = np.array([float(x) for x in z0.xpath('//econst/text()')])
fig,ax = plt.subplots()
ax.plot(np0, label='etotal')
ax.plot(np1, label='econst')
ax.grid()
