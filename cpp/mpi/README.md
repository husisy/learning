# Message Passing Interface

1. link
   * [MPI tutorial](https://mpitutorial.com/)
   * [MPICH](https://www.mpich.org/)
   * [github - MS MPI](https://github.com/microsoft/Microsoft-MPI)
2. `mpi.h`
   * `/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h`
   * `/opt/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64/include/mpi.h`
3. `CPLUS_INCLUDE_PATH`
4. MPI datatype
5. `MPI_Status`: `MPI_STATUS_IGNORE`
6. 常用函数
   * `MPI_Init() MPI_Finalize() MPI_Abort()`
   * `MPI_Comm_rank(MPI_COMM_WORLD, int*)`
   * `MPI_Comm_size(MPI_COMM_WORLD, int*)`
   * `MPI_Send() MPI_Recv()`
   * `MPI_Get_count() MPI_Probe()`

| MPI datatype | C equivalent |
| :-: | :-: |
| `MPI_SHORT` | `short int` |
| `MPI_INT` | `int` |
| `MPI_LONG` | `long int` |
| `MPI_LONG_LONG` | `long long int` |
| `MPI_UNSIGNED_CHAR` | `unsigned char` |
| `MPI_UNSIGNED_SHORT` | `unsigned short int` |
| `MPI_UNSIGNED` | `unsigned int` |
| `MPI_UNSIGNED_LONG` | `unsigned long int` |
| `MPI_UNSIGNED_LONG_LONG` | `unsigned long long int` |
| `MPI_FLOAT` | `float` |
| `MPI_DOUBLE` | `double` |
| `MPI_LONG_DOUBLE` | `long double` |
| `MPI_BYTE` | `char` |

```bash
sudo apt update
sudo apt install mpich
dpkg -L mpich
mpichversion
mpirun.mpich --version
mpiexec.mpich
```

TODO

1. host文件配置 `MPI_HOSTS`

## ws00

1. 编译 `mpicxx -o tbd00.exe draft00.cpp`
   * `mpicxx.mpich -o tbd00.exe draft00.cpp`
2. 运行 `mpirun -n 2 ./tbd00.exe`
   * `mpirun.mpich -n 2 ./tbd00.exe`
   * `./tbd00.exe`
