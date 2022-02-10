# mpi4py

1. link
   * [documentation](https://mpi4py.readthedocs.io/en/stable/)
   * [github-mpi4py](https://github.com/mpi4py/mpi4py)
   * [misc-tutorial](https://rabernat.github.io/research_computing/parallel-programming-with-mpi-for-python.html)
   * [ipyparallel documentation](https://ipyparallel.readthedocs.io/en/latest/)
   * [ZeroMQ official site](https://zeromq.org/)
2. open source MPI implementation
   * [MPICH](https://www.mpich.org/)
   * [Open MPI](https://www.open-mpi.org/)
   * [github-PyPar](https://github.com/daleroberts/pypar): Efficient and scalable parallelism using the message passing interface (MPI) to handle big data and highly computational problems
   * [github-pyMPI](https://github.com/dopefishh/pympi): A python module for processing ELAN and Praat annotation files
3. install
   * `conda install -c conda-forge mpi4py`
   * `pip install mpi4py`
4. download [MS-MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467)
   * `msmpisdk.smi`
   * `msmpisetup.exe`
   * see [stackoverflow-MPI-DLL-error](https://stackoverflow.com/a/57781714)
5. concept: heterogeneous computing environments
6. generic python objects and buffer-like objects
   * [python-buffer protocol](https://docs.python.org/3/c-api/buffer.html)
   * [PEP3118 revising the buffer protocol](https://www.python.org/dev/peps/pep-3118/)
7. 偏见
   * mpi4py能够在`python/ipython`交互环境下运行，`rank=0`，**仅**用于测试

```Python
MPI.Comm
MPI.Intracommm
MPI.Intercomm
MPI.Comm.Is_inter
MPI.Comm.Is_intra
MPI.COMM_SELF
MPI.COMM_WORLD
```

```bash
mpiexec -n 4 python draft00.py
```

## openmpi

1. link
2. installation: `conda install -c conda-forge openmpi`
3. 关键环境变量
   * `OMPI_COMM_WORLD_SIZE`: world size
   * `OMPI_COMM_WORLD_RANK`: rank
   * `OMPI_COMM_WORLD_LOCAL_RANK`: local rank

> For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default.
> To enable it, please set the environmental variable OMPI_MCA_opal_cuda_support=true before
> launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:
> mpiexec --mca opal_cuda_support 1 ...'

| key | proc-0 | proc-1 | proc-2 |
| :-: | :-: | :-: | :-: |
| `OMPI_MCA_ess_base_jobid` | `2404974593` | `2404974593` | `2404974593` |
| `OMPI_MCA_initial_wdir` | `/home/zhangc/learning/python/openmpi` | `/home/zhangc/learning/python/openmpi` | `/home/zhangc/learning/python/openmpi` |
| `OMPI_FILE_LOCATION` | `/tmp/ompi.p720.1000/pid.29162/0/0` | `/tmp/ompi.p720.1000/pid.29162/0/0` | `/tmp/ompi.p720.1000/pid.29162/0/0` |
| `PMIX_SYSTEM_TMPDIR` | `/tmp` | `/tmp` | `/tmp` |
| `PMIX_DSTORE_21_BASE_PATH` | `/tmp/ompi.p720.1000/pid.29162/pmix_dstor_ds21_29162` | `/tmp/ompi.p720.1000/pid.29162/pmix_dstor_ds21_29162` | `/tmp/ompi.p720.1000/pid.29162/pmix_dstor_ds21_29162` |
| `OMPI_COMM_WORLD_SIZE` | `3` | `3` | `3` |
| `OMPI_COMMAND` | `python` | `python` | `python` |
| `OMPI_MCA_orte_launch` | `1` | `1` | `1` |
| `OMPI_FIRST_RANKS` | `0` | `0` | `0` |
| `OMPI_MCA_orte_tmpdir_base` | `/tmp` | `/tmp` | `/tmp` |
| `OMPI_MCA_orte_local_daemon_uri` | `2404974592.0;tcp://192.168.136.31,172.17.0.1:60877` | `2404974592.0;tcp://192.168.136.31,172.17.0.1:60877` | `2404974592.0;tcp://192.168.136.31,172.17.0.1:60877` |
| `PMIX_GDS_MODULE` | `ds21,ds12,hash` | `ds21,ds12,hash` | `ds21,ds12,hash` |
| `OMPI_MCA_ess_base_vpid` | `0` | `1` | `2` |
| `PMIX_DSTORE_ESH_BASE_PATH` | `/tmp/ompi.p720.1000/pid.29162/pmix_dstor_ds12_29162` | `/tmp/ompi.p720.1000/pid.29162/pmix_dstor_ds12_29162` | `/tmp/ompi.p720.1000/pid.29162/pmix_dstor_ds12_29162` |
| `PMIX_VERSION` | `3.1.5` | `3.1.5` | `3.1.5` |
| `PMIX_SERVER_TMPDIR` | `/tmp/ompi.p720.1000/pid.29162` | `/tmp/ompi.p720.1000/pid.29162` | `/tmp/ompi.p720.1000/pid.29162` |
| `OMPI_MCA_orte_ess_num_procs` | `3` | `3` | `3` |
| `PMIX_RANK` | `0` | `1` | `2` |
| `OMPI_MCA_pmix` | `^s1,s2,cray,isolated` | `^s1,s2,cray,isolated` | `^s1,s2,cray,isolated` |
| `OMPI_ARGV` | `demo_mpi_environ.py --launch` | `demo_mpi_environ.py --launch` | `demo_mpi_environ.py --launch` |
| `OMPI_MCA_orte_precondition_transports` | `99d9327765c60795-36cdb2577c5392a7` | `99d9327765c60795-36cdb2577c5392a7` | `99d9327765c60795-36cdb2577c5392a7` |
| `HFI_NO_BACKTRACE` | `1` | `1` | `1` |
| `OMPI_MCA_orte_hnp_uri` | `2404974592.0;tcp://192.168.136.31,172.17.0.1:60877` | `2404974592.0;tcp://192.168.136.31,172.17.0.1:60877` | `2404974592.0;tcp://192.168.136.31,172.17.0.1:60877` |
| `OMPI_MCA_orte_top_session_dir` | `/tmp/ompi.p720.1000` | `/tmp/ompi.p720.1000` | `/tmp/ompi.p720.1000` |
| `PMIX_ID` | `2404974593.0` | `2404974593.1` | `2404974593.2` |
| `PMIX_SERVER_URI3` | `2404974592.0;tcp4://127.0.0.1:53851` | `2404974592.0;tcp4://127.0.0.1:53851` | `2404974592.0;tcp4://127.0.0.1:53851` |
| `OMPI_MCA_mpi_oversubscribe` | `0` | `0` | `0` |
| `OMPI_MCA_orte_ess_node_rank` | `0` | `1` | `2` |
| `OMPI_MCA_orte_num_nodes` | `1` | `1` | `1` |
| `IPATH_NO_BACKTRACE` | `1` | `1` | `1` |
| `OMPI_COMM_WORLD_RANK` | `0` | `1` | `2` |
| `PMIX_BFROP_BUFFER_TYPE` | `PMIX_BFROP_BUFFER_NON_DESC` | `PMIX_BFROP_BUFFER_NON_DESC` | `PMIX_BFROP_BUFFER_NON_DESC` |
| `OMPI_MCA_ess` | `^singleton` | `^singleton` | `^singleton` |
| `OMPI_MCA_orte_bound_at_launch` | `1` | `1` | `1` |
| `OMPI_NUM_APP_CTX` | `1` | `1` | `1` |
| `PMIX_HOSTNAME` | `p720` | `p720` | `p720` |
| `PMIX_PTL_MODULE` | `tcp,usock` | `tcp,usock` | `tcp,usock` |
| `OMPI_COMM_WORLD_LOCAL_RANK` | `0` | `1` | `2` |
| `PMIX_SERVER_URI2` | `2404974592.0;tcp4://127.0.0.1:53851` | `2404974592.0;tcp4://127.0.0.1:53851` | `2404974592.0;tcp4://127.0.0.1:53851` |
| `OMPI_APP_CTX_NUM_PROCS` | `3` | `3` | `3` |
| `OMPI_COMM_WORLD_NODE_RANK` | `0` | `1` | `2` |
| `OMPI_UNIVERSE_SIZE` | `24` | `24` | `24` |
| `OMPI_MCA_orte_app_num` | `0` | `0` | `0` |
| `PMIX_SECURITY_MODE` | `native` | `native` | `native` |
| `PMIX_MCA_mca_base_component_show_load_errors` | `1` | `1` | `1` |
| `PMIX_NAMESPACE` | `2404974593` | `2404974593` | `2404974593` |
| `OMPI_MCA_shmem_RUNTIME_QUERY_hint` | `mmap` | `mmap` | `mmap` |
| `OMPI_COMM_WORLD_LOCAL_SIZE` | `3` | `3` | `3` |
| `PMIX_SERVER_URI21` | `2404974592.0;tcp4://127.0.0.1:53851` | `2404974592.0;tcp4://127.0.0.1:53851` | `2404974592.0;tcp4://127.0.0.1:53851` |
| `OMPI_MCA_orte_jobfam_session_dir` | `/tmp/ompi.p720.1000/pid.29162` | `/tmp/ompi.p720.1000/pid.29162` | `/tmp/ompi.p720.1000/pid.29162` |
