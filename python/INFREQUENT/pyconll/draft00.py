import os
from pyconll import load_from_string, iter_from_string, load_from_file, iter_from_file

hf_data = lambda *x: os.path.join('data', *x)

with open(hf_data('basic.conll'), encoding='utf-8') as fid:
    z1 = load_from_string(fid.read())
# z1 = load_from_file(hf_data('basic.conll'))
x1 = z1[0]
# print(x1.conll())

# [x.id for x in x1]
# '\n'.join(x.conll() for x in x1)
# x1['10'].form
# x1[9].form
