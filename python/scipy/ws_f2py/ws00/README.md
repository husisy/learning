# scipy-f2py-mwe00

```bash
# f2py -m my_f2py_module my_f2py_module.f
# apt install gfortran
f2py -m my_f2py_module -c my_f2py_module.f
# python -m numpy.f2py -m my_f2py_module -c my_f2py_module.f
pytest .
```
