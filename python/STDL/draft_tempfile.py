import os
import tempfile

z0 = tempfile.TemporaryFile()
z0.write('233'.encode())
z0.fileno()
z0.close()

z0 = tempfile.TemporaryDirectory()
z0.name
tmp0 = os.path.join(z0.name, 'tbd00.txt')
with open(tmp0, 'w') as fid:
    fid.write('233')
with open(tmp0, 'r') as fid:
    fid.read()
z0.cleanup()

with tempfile.TemporaryDirectory() as z0:
    tmp0 = os.path.join(z0, 'tbd00.txt')
