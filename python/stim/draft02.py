# sinter fail on macos, pass on linux
import os
import typing
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import stim
import pymatching
import sinter

## repitition code
distance_list = [3, 5, 7, 9]
physical_error_list = [0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
hf0 = lambda d,p: stim.Circuit.generated("repetition_code:memory", rounds=d*3, distance=d, before_round_data_depolarization=p)
tasks = [sinter.Task(circuit=hf0(d, p), json_metadata={'d':d, 'p':p}) for d in distance_list for p in physical_error_list]
collected_stats = sinter.collect(num_workers=4, tasks=tasks, decoders=['pymatching'], max_shots=100_000, max_errors=500)
collected_stats[0]
# sinter.TaskStats(strong_id='xxx', decoder='pymatching', json_metadata={'d': 3, 'p': 0.08}, shots=7441, errors=522, seconds=0.008032)

fig, ax = plt.subplots(1, 1)
sinter.plot_error_rate(ax=ax, stats=collected_stats,
        x_func=lambda x: x.json_metadata['p'], group_func=lambda x: x.json_metadata['d'])
ax.set_ylim(1e-4, 1e-0)
ax.set_xlim(5e-2, 5e-1)
ax.loglog()
ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
ax.set_xlabel("Phyical Error Rate")
ax.set_ylabel("Logical Error Rate per Shot")
ax.grid(which='both')
ax.legend()
fig.savefig('tbd00.png', dpi=200)



## https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb
distance_list = [3, 5, 7]
physical_error_list = [0.008, 0.009, 0.01, 0.011, 0.012]
task_list = []
for d in distance_list:
    for p in physical_error_list:
        circ = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=d * 3, distance=d, after_clifford_depolarization=p,
                after_reset_flip_probability=p, before_measure_flip_probability=p, before_round_data_depolarization=p)
        task_list.append(sinter.Task(circuit=circ, json_metadata={'d':d, 'r':d*3, 'p':p}))
# save_resume_filepath
num_worker = os.cpu_count()
data = sinter.collect(num_workers=num_worker, tasks=task_list,
        decoders=['pymatching'], max_shots=1000000, max_errors=5000, print_progress=True)

fig, ax = plt.subplots()
sinter.plot_error_rate(ax=ax, stats=data,
    x_func=lambda x: x.json_metadata['p'],
    group_func=lambda x: x.json_metadata['d'],
    failure_units_per_shot_func=lambda x: x.json_metadata['r'], #per round error rates instead of per shot error rates
)
ax.set_ylim(5e-3, 5e-2)
ax.set_xlim(0.008, 0.012)
ax.loglog()
ax.set_title("Surface Code Error Rates per Round under Circuit Noise")
ax.set_xlabel("Phyical Error Rate")
ax.set_ylabel("Logical Error Rate per Round")
ax.grid(which='major')
ax.grid(which='minor')
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


noise = 1e-3
distance_list = [3, 5, 7, 9]
task_list = []
for d in distance_list:
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=d*3, distance=d,
            after_clifford_depolarization=noise, after_reset_flip_probability=noise,
            before_measure_flip_probability=noise, before_round_data_depolarization=noise)
    task_list.append(sinter.Task(circuit=circ, json_metadata={'d':d, 'r':d*3, 'p':noise}))
num_worker = os.cpu_count()
data = sinter.collect(num_workers=num_worker, tasks=task_list,
    decoders=['pymatching'], max_shots=5000000, max_errors=100, print_progress=True)

assert all(x.errors>0 for x in data)
xs = np.array([x.json_metadata['d'] for x in data])
ys = np.array([sinter.shot_error_rate_to_piece_error_rate(x.errors/x.shots, pieces=x.json_metadata['r']) for x in data])
fit = scipy.stats.linregress(xs, np.log(ys)) #linear fit in log space
fig, ax = plt.subplots()
ax.scatter(xs, ys, label=f"sampled logical error rate at p={noise}")
tmp0 = np.array([0, 25])
tmp1 = np.exp(fit.intercept + fit.slope*tmp0)
ax.plot(tmp0, tmp1, linestyle='--', label='least squares line fit')
ax.set_ylim(1e-12, 1e-0)
ax.set_xlim(0, 25)
ax.semilogy()
ax.set_title("Projecting distance needed to survive a trillion rounds")
ax.set_xlabel("Code Distance")
ax.set_ylabel("Logical Error Rate per Round")
ax.grid(which='major')
ax.grid(which='minor')
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
