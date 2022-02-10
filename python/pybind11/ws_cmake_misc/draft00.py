from zctest_pybind11._cpp import Pet, new_pet, print_dict
from zctest_pybind11._cpp import utf8_string, utf8_charptr, return_utf8_string, return_utf8_charptr

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-5: round(hfe(x,y,eps),5)

z0 = Pet('233')
print(z0)
z0.get_name()
z0.set_name('abb')
z0.name = '233'

print(new_pet())

print_dict({"1": "-1", "233":"-233"})

utf8_string('233')
utf8_string(b'233')
utf8_charptr('233')
utf8_charptr(b'233')
print(return_utf8_string())
print(return_utf8_charptr())


import numpy as np
from zctest_pybind11._cpp import CArray, c_sum, c_sum_complex

# CArray -> np
z0 = CArray(3)
z1 = np.array(z0, copy=False)
print('before: ', z0.get_info())
z1[1] = 233
print('after: ', z0.get_info())

# np -> CArray
np0 = np.random.rand(3)
z0 = CArray(np0, copy=False)

# np(strides!=size) -> CArray
np0 = np.random.rand(3) + np.random.rand(3)*1j
print('hfe(real): ', hfe(np0.real, CArray(np0.real)))
print('hfe(imag): ', hfe(np0.imag, CArray(np0.imag)))

np0 = np.random.rand(5) + np.random.rand(5)*1j
print('hfe(c_sum_complex): ', hfe(np0.sum(), c_sum_complex(np0)))
