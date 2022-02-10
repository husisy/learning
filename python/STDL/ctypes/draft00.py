import ctypes
import numpy as np

myclib = ctypes.CDLL('tbd00/cextension.so')
myclib.hf0()
myclib.hf1(233)

npclib = np.ctypeslib.load_library('cextension.so', 'tbd00')
npclib.hf2.argtypes = [np.ctypeslib.ndpointer(np.float64, flags='aligned, c_contiguous'), ctypes.c_int]
np0 = np.array([2,3,3], dtype=np.float64)
npclib.hf2(np0, np0.size)

mycpplib = ctypes.CDLL('tbd00/cppextension.so')
mycpplib.hf1(233)

# windows only
ctypes.windll.kernel32
ctypes.windll.user32
ctypes.cdll.msvcrt
# libc = ctypes.windll.msvcrt
libc = ctypes.cdll.msvcrt

# linux only
libc = ctypes.CDLL('libc.so.6')
libc.printf

libc.time(None)
libc.printf(b'hello world\n')

#ctypes
ctypes.c_int(0)
ctypes.c_ushort(-3) #65533
x = ctypes.c_wchar_p('hell world')
x.value

