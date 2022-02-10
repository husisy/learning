import time

import my_swig_pkg00


assert 3.0 == my_swig_pkg00.cvar.My_variable
assert 120 == my_swig_pkg00.fact(5)

py_time = time.time()
py_time = time.localtime(py_time)
py_time = time.asctime(py_time)
c_time = my_swig_pkg00.get_time().strip()
assert py_time == c_time

assert 1 == my_swig_pkg00.my_mod(7, 3)

result = my_swig_pkg00.echo('message')
assert isinstance(result, str)
assert 'message' == result

data = my_swig_pkg00.vector_int2str([1, 2, 3])
assert isinstance(data, tuple)
assert ('1', '2', '3') == data


z0 = my_swig_pkg00._stl_example.new_Str2intMap()
tmp0 = {'1':1, '2':2, '3':3}
for k,v in tmp0.items():
    my_swig_pkg00._stl_example.Str2intMap___setitem__(z0, k, v)
    assert my_swig_pkg00._stl_example.Str2intMap___getitem__(z0, k)==v
z1 = my_swig_pkg00._stl_example.reverse_map(z0)
for k,v in tmp0.items():
    assert z1[v]==k
my_swig_pkg00._stl_example.delete_Str2intMap(z0)
