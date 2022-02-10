import os
import pickle
import argparse
import subprocess

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

# python demo_mpi_environ.py
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true') #True if supplied, otherwise False
    parser.add_argument('--worker', default=3, type=int)
    args = parser.parse_args()
    if args.launch:
        rank = os.environ.get('OMPI_COMM_WORLD_RANK')
        assert rank is not None
        with open(hf_file(f'demo_mpi_environ{rank}.pkl'), 'wb') as fid:
            pickle.dump(dict(**os.environ), fid)
    else:
        tmp0 = ['mpirun', '-n', str(args.worker), 'python', 'demo_mpi_environ.py', '--launch']
        subprocess.run(tmp0)
        all_environ = [dict(**os.environ)]
        for ind0 in range(args.worker):
            with open(hf_file(f'demo_mpi_environ{ind0}.pkl'), 'rb') as fid:
                all_environ.append(pickle.load(fid))
        all_key = {y for x in all_environ for y in x.keys()}
        all_key_value = {x:[y.get(x,None) for y in all_environ] for x in all_key}
        diff_key_value = {k:v for k,v in all_key_value.items() if len(set(v))>1}
        if all(x[0] is None for x in diff_key_value.values()):
            diff_key_value = {k:v[1:] for k,v in diff_key_value.items()}
        for k,v in diff_key_value.items():
            tmp0 = ' | '.join([f'`{x}`' for x in v])
            print(f'| `{k}` | {tmp0} |')
