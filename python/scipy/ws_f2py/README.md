# workspace-f2py

1. link
   * [documentation/f2py](https://numpy.org/doc/stable/user/c-info.python-as-glue.html#f2py)

usage

```bash
f2py -m add add.f
# apt install gfortran
f2py -c -m add add.f
```

```Python
import numpy as np
import numpy.f2py
np.f2py
```
