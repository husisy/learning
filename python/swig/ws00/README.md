# ws00

```bash
swig -python example.i
gcc -fPIC -c example.c example_wrap.c -I/path/to/miniconda3/envs/cuda111/include/python3.9
ld -shared example.o example_wrap.o -o _example.so
```

```Python
import example
```
