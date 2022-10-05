# workspace-f2py

1. link
   * [documentation/f2py](https://numpy.org/doc/stable/user/c-info.python-as-glue.html#f2py)

usage

```bash
# f2py -m my_f2py_module my_f2py_module.f
# apt install gfortran
f2py -m my_f2py_module -c my_f2py_module.f
# python -m numpy.f2py
```

```Python
import numpy as np
import numpy.f2py
np.f2py
```
