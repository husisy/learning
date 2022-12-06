import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import z2pack
import tbmodels

def hamiltonian(k):
    kx, ky, kz = k
    ret = np.array([[kz, kx - 1j * ky], [kx + 1j * ky, -kz]])
    return ret

system = z2pack.hm.System(hamiltonian)


# model = tbmodels.Model.from_wannier_files(hr_file='path_to_directory/wannier90_hr.dat')
# system = z2pack.tb.System(model)



import logging
import itertools



logging.getLogger('z2pack').setLevel(logging.WARNING)

t1, t2 = (0.2, 0.3)

settings = {
    'num_lines': 11,
    'pos_tol': 1e-2,
    'gap_tol': 2e-2,
    'move_tol': 0.3,
    'iterator': range(8, 27, 2),
    'min_neighbour_dist': 1e-2,
}

model = tbmodels.Model(
    on_site=(1, 1, -1, -1),
    pos=[[0., 0., 0.], [0., 0., 0.], [0.5, 0.5, 0.], [0.5, 0.5, 0.]],
    occ=2
)

for p, R in zip([1, 1j, -1j, -1], itertools.product([0, -1], [0, -1], [0])):
    model.add_hop(overlap=p * t1, orbital_1=0, orbital_2=2, R=R)
    model.add_hop(overlap=p.conjugate() * t1, orbital_1=1, orbital_2=3, R=R)

for r in itertools.permutations([0, 1]):
    R = r[0],r[1],0
    model.add_hop(t2, 0, 0, R)
    model.add_hop(t2, 1, 1, R)
    model.add_hop(-t2, 2, 2, R)
    model.add_hop(-t2, 3, 3, R)

tb_system = z2pack.tb.System(model)

result = z2pack.surface.run(system=tb_system, surface=lambda s, t: [s / 2., t, 0], **settings)

fig, ax = plt.subplots()
z2pack.plot.wcc(result, axis=ax)
fig.savefig('tbd00.png')

print("t1: {0}, t2: {1}, Z2 invariant: {2}".format(t1, t2, z2pack.invariant.z2(result)))




import os
import lzma
import pickle
from contextlib import suppress

import z2pack
from tbmodels import Model
import matplotlib.pyplot as plt

MODEL_NAME = 'wte2_soc'
MODEL_SOURCE = os.path.join('data', MODEL_NAME + '.json')
MODEL_PATH = os.path.join('data', MODEL_NAME + '.p')

# creating the necessary subfolders
subfolders = ['results', 'plots']
for s in subfolders:
    with suppress(FileExistsError):
        os.mkdir(s)


def calculate_chirality(tag, center, radius, overwrite=False, **kwargs):
    # converting the Model to the pickle format (which is quicker to load)
    # Note that keeping only the pickle format is dangerous, because it
    # may become unreadable -- use the JSON format for long-term saving.
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except IOError:
        # The .xz compression is used to avoid the GitHub file size limit
        with lzma.open(MODEL_SOURCE + '.xz') as fin, open(MODEL_SOURCE, 'wb') as fout:
            fout.write(fin.read())

        model = Model.from_json_file(MODEL_SOURCE)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    system = z2pack.tb.System(model)
    full_name = MODEL_NAME + '_' + tag
    res = z2pack.surface.run(
        system=system,
        surface=z2pack.shape.Sphere(center, radius),
        save_file=os.path.join('results', full_name + '.p'),
        load=not overwrite,
        **kwargs
    )
    # Again, the pickle format is used because it is faster than JSON
    # or msgpack. If you wish to permanently store the result, use
    # z2pack.io.save(res, os.path.join('results', full_name + '.json'))
    print('Chern number:', z2pack.invariant.chern(res))


def plot_chirality(tag, ax):
    full_name = MODEL_NAME + '_' + tag
    res = z2pack.io.load(os.path.join('results', full_name + '.p'))
    z2pack.plot.chern(res, axis=ax)


calculate_chirality('0', [0.1203, 0.05232, 0.], 0.005)
calculate_chirality(
    '1', [0.1211, 0.02887, 0.], 0.005, iterator=range(10, 33, 2)
)

# plot
fig, ax = plt.subplots(
    1, 2, figsize=[4, 2], sharey=True, gridspec_kw=dict(wspace=0.3)
)
ax[0].set_xticks([0, 1])
ax[1].set_xticks([0, 1])
ax[0].set_xticklabels([r'$-\pi$', r'$0$'])
ax[1].set_xticklabels([r'$-\pi$', r'$0$'])
ax[0].set_yticks([0, 1])
ax[1].set_yticks([0, 1])
ax[1].set_yticklabels([r'$0$', r'$2\pi$'])
ax[0].set_xlabel(r'$\theta$')
ax[1].set_xlabel(r'$\theta$')
ax[0].set_ylabel(r'$\bar{\varphi}$', rotation='horizontal')
ax[0].text(-0.2, 1.05, r'(a)', ha='right')
ax[1].text(-0.05, 1.05, r'(b)', ha='right')
plot_chirality('0', ax[0])
plot_chirality('1', ax[1])
plt.savefig(
    'plots/WTe2_chirality.pdf', bbox_inches='tight', rasterized=True
)
