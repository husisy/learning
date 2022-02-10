import os
import pickle
import argparse
import subprocess

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

# python demo_launch_environ.py
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true') #True if supplied, otherwise False
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--local_rank', default=None, type=int, help='provided by torch.distributed.launch')
    args = parser.parse_args()
    if args.launch:
        assert args.local_rank is not None
        tmp0 = os.environ.get('RANK')
        print(f'[local_rank={args.local_rank}] RANK={tmp0}')
        with open(hf_file(f'demo_lauch_environ{args.local_rank}.pkl'), 'wb') as fid:
            pickle.dump(dict(**os.environ), fid)
    else:
        tmp0 = ['python', '-m', 'torch.distributed.launch', f'--nproc_per_node={args.worker}', 'demo_launch_environ.py', '--launch']
        subprocess.run(tmp0)
        origin_environ = dict(**os.environ)
        for ind0 in range(args.worker):
            with open(hf_file(f'demo_lauch_environ{ind0}.pkl'), 'rb') as fid:
                z0 = pickle.load(fid)
            all_key = set(origin_environ.keys()) | set(z0.keys())
            diff_key = [k for k in all_key if origin_environ.get(k)!=z0.get(k)]
            for key in diff_key:
                print(f'key={key}')
                print(f'\t origin: {origin_environ.get(key)}')
                print(f'\t proc-{ind0}: {z0.get(key)}')
