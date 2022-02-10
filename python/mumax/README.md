# mumax

1. link
   * [github](https://github.com/mumax/3)
   * [documentation](https://mumax.github.io/)
   * [paper](https://aip.scitation.org/doi/10.1063/1.4899186) The design and verification of MuMax3
   * [OOMMF](https://math.nist.gov/oommf/) the Object Oriented MicroMagnetic Framework project
2. install（见下方代码片段）

```bash
wget https://mumax.ugent.be/mumax3-binaries/mumax3.10_linux_cuda11.0.tar.gz
untar mumax3.10_linux_cuda11.0.tar.gz
export PATH="xxx:$PATH"
```

server-client运行

1. 运行命令`mumax3-server`
2. 浏览器访问`http://127.0.0.1:35360`
   * 端口可能会边，以`mumax3-server`输出信息为准

## ws00

```bash
mumax3 draft00.mx3
```
