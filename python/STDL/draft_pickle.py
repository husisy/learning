import os
import pickle

hf_file = lambda *x: os.path.join('tbd00', *x)

tmp1 = {y:x for x,y in enumerate('abcde')}
tmp2 = pickle.loads(pickle.dumps(tmp1))

with open(hf_file('tbd01.pkl'), 'wb') as fid:
    pickle.dump(tmp1, fid)
with open(hf_file('tbd01.pkl'), 'rb') as fid:
    _ = pickle.load(fid)
