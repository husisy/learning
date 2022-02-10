# ITensor

1. link
   * [official site](https://itensor.org/index.html)
   * [github / itensor](https://github.com/ITensor)
2. install
   * `sudo apt install libblas-dev liblapack-dev` install blas and lapack, see [link](https://askubuntu.com/a/736684)
   * see [documentation / install](https://itensor.org/docs.cgi?vers=cppv3&page=install)
3. terminology: Abelian symmetry, non-Abelian symmetry

```bash
g++ draft00.cpp -o tbd00.exe -m64 -std=c++17 -fconcepts -fPIC -I"/home/zhangc/software/itensor" -O2 -DNDEBUG -Wall -Wno-unknown-pragmas -L"/home/zhangc/software/itensor/lib" -litensor -lpthread -L/usr/lib -lblas -llapack
```
