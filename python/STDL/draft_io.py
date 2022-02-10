import os
from io import StringIO, BytesIO

hf_data = lambda *x: os.path.join('data', *x)
assert os.path.exists(hf_data())
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

# file IO
with open(hf_data('example00.txt'), 'r') as fid:
    # print(fid.read())
    print(''.join(x.strip('\n') for x in fid))

with open(hf_data('example01_cn.txt'), 'r', encoding='utf-8') as fid:
    _ = fid.read()

with open(hf_file('tbd01.txt'), 'w') as fid:
    fid.write('hello\n \nword\n!')

# string / byte IO
sid = StringIO()
sid.write('hello')
sid.write(' ')
sid.write('word!')
sid.getvalue()

sid = StringIO('hello\n \nworld!\n')
''.join(x.strip('\n') for x in sid.readlines())

sid = BytesIO()
sid.write('hello world!'.encode('utf-8'))
print(sid.getvalue().decode())

sid = BytesIO('hello world!'.encode('utf-8'))
print(sid.read().decode('utf-8'))
