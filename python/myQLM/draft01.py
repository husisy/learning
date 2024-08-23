import argparse
import time
import numpy as np

import qat.core
import qat.lang

np_rng = np.random.default_rng()

def hf_task(num_qubit, num_depth, num_repeat, seed):
    # https://myqlm.github.io/01_getting_started/04_variational.html
    # antiferromagnetic Heisenberg Hamiltonian
    np_rng = np.random.default_rng(seed)
    tmp0 = list(range(0, num_qubit-1, 2)) + list(range(1, num_qubit-1, 2))
    num_theta = num_depth * len(tmp0) * 4

    @qat.lang.qfunc(thetas=num_theta)
    def energy(thetas):
        tmp0 = list(range(0, num_qubit-1, 2)) + list(range(1, num_qubit-1, 2))
        thetas = thetas.reshape(num_depth, len(tmp0), 4)
        for ind0 in range(num_depth):
            for ind1 in range(len(tmp0)):
                qat.lang.RY(thetas[ind0,ind1,0])(tmp0[ind1])
                qat.lang.RY(thetas[ind0,ind1,1])(tmp0[ind1]+1)
                qat.lang.RZ(thetas[ind0,ind1,2])(tmp0[ind1])
                qat.lang.RZ(thetas[ind0,ind1,3])(tmp0[ind1]+1)
                qat.lang.CNOT(tmp0[ind1], tmp0[ind1]+1)
        pauli = qat.core.Observable
        hf0 = lambda a,b: pauli.sigma_z(a) * pauli.sigma_z(b) + pauli.sigma_x(a) * pauli.sigma_x(b) + pauli.sigma_y(a) * pauli.sigma_y(b)
        obs = sum([hf0(x,x+1) for x in range(num_qubit-1)])
        return obs

    theta0 = np_rng.uniform(0, 2*np.pi, num_theta)
    time_list = []
    value_list = []
    for _ in range(num_repeat+1):
        t0 = time.time()
        value_list.append(energy(theta0))
        time_list.append(time.time() - t0)
    time_list = np.array(time_list[1:])
    value_list = np.array(value_list[1:])
    # import jax
    # z0 = energy(jax.device_put(theta0))
    # result = energy.run()
    # print(f"Minimum VQE energy = {result.value}")
    return time_list, value_list

# python draft00.py test --num_qubit_list 10,12,14,16,18,20,22,24 --num_depth=5
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='cupy benchmark')
    parser.add_argument('key', type=str, help='unique device id, e.g. dgx-station, cdcloud-dcu', default='test')
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--num_qubit_list', type=str, default='10,12,14,16,18,20,22,24,26,28')
    # 28qubits almost 100s, need 4GB memory per state
    parser.add_argument('--num_depth', type=int, default=5)
    parser.add_argument('--num_repeat', type=int, default=3)
    args = parser.parse_args()
    args.num_qubit_list = [int(x) for x in args.num_qubit_list.split(',')]

    print('key:', args.key)
    for num_qubit in args.num_qubit_list:
        time_list, value_list = hf_task(num_qubit, args.num_depth, args.num_repeat, args.seed)
        tmp0 = ','.join([str(x) for x in time_list])
        tmp1 = ','.join([str(x) for x in value_list])
        print(f"#num_qubit={num_qubit}\ntime: {tmp0}\nvalue: {tmp1}")



def draw_time_usage():
    import numpy as np
    import matplotlib.pyplot as plt
    qubit_list = [8,10,12,14,16,18,20,22,24,26]
    time_list = [
        [0.04625463, 0.04564381, 0.04594707],
        [0.05856133, 0.05822396, 0.057971],
        [0.07127929, 0.0720253, 0.0719862],
        [0.08554411, 0.08499765, 0.08519173],
        [0.10245371, 0.10148191, 0.10178113],
        [0.13299823, 0.13378811, 0.13487363],
        [0.25141072, 0.2555747,  0.25324798],
        [3.52684593, 3.49917006, 3.54641795],
        [3.08366299, 3.09551597, 3.09078908],
        [12.81701064, 12.8663528,  12.86538959],
    ]
    fig,ax = plt.subplots()
    ax.plot(qubit_list, np.array(time_list).mean(axis=1), 'x-')
    ax.grid()
    ax.set_xlabel('#qubits')
    ax.set_ylabel('#time (s)')
    ax.set_title('Aspen/myQLM VQE time usage')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)
