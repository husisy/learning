# slurm作业调度系统

1. link
   * [official site](https://www.schedmd.com/index.php)
   * [英文文档](https://slurm.schedmd.com/documentation.html)
   * [gitbook/中文文档](https://docs.slurm.cn/users/)
2. Simple Linux Utility for Resource Management (SLURM)
3. 每个计算节点上运行的`slurmd`守护程序
4. 在管理节点上运行的中央`slurmctld`守护程序
5. 用户命令：`sacct salloc sattach sbatch sbcast scancel scontrol sinfo SMAP SQUEUE SRUN strigger sview`
6. concept: node, partition, job, job step
7. `sinfo` 分区和节点的状态
   * partition：按照state分为多行显示
   * avail: `up`, `down`
   * state: `down*`表示节点无响应
   * nodelist
8. `squeue` 工作或工作步骤的状态
   * jobid
   * partition
   * name
   * user
   * ST: `R=Running`, `PD=Pending`
   * time
   * nodes
   * nodelist(reason)
   * `squeue --user=xxx`, `squeue -u=xxx`
9. `scontrol`
   * `scontrol show partition`
   * `scontrol show node hhnode-ib-146`
   * `scontrol show job`
10. `srun`
    * `srun --partition=xxx --nodes=3 --label /bin/hostname`, `srun -p xxx -N3 -l /bin/hostname`
    * `srun --partition=xxx --ntasks=3 --label /bin/hostname`
11. 文件系统: NFS, Lustre

```bash
pip install proxy.py
proxy --hostname 127.0.0.1 --port 23333
ssh -R 127.0.0.1:23333:127.0.0.1:23333 cqc-cluster
export http_proxy=http://127.0.0.1:23333
export https_proxy=http://127.0.0.1:23333
```

```bash
docker pull xenonmiddleware/slurm:latest
docker run -it --rm xenonmiddleware/slurm bash
```

## mwe

`my.script`

```bash
#!/bin/sh
#SBATCH --time=1
/bin/hostname
srun -l /bin/hostname
srun -l /bin/pwd
```

1. `sbatch -p xxx -n4 -o my.stdout my.script`

## OpenPBS

1. link
   * [wiki/Portable Batch System](https://en.wikipedia.org/wiki/Portable_Batch_System)
   * [githubOpenPBS](https://github.com/openpbs/openpbs)
   * [OpenPBS/official site](https://www.openpbs.org/)
   * [openpbs/documentation](https://www.altair.com/pbs-works-documentation/)
