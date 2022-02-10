# k3d

1. link
   * [github](https://github.com/K3D-tools/K3D-jupyter)
2. install
   * 安装nodejs和npm [link](https://nodejs.org/en/download/)，两者同一个安装文件
   * **不建议**使用conda安装`nodejs`：`nodejs`版本过低，无法安装`npm`
   * `conda install -c conda-forge k3d`
   * `pip install k3d`
   * 配置jupyter-notebook（见下）

```bash
# jupyter lab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install k3d

# jupyter notebook (no need if use jupyter lab only)
jupyter nbextension install --py --sys-prefix k3d
jupyter nbextension enable k3d --py --sys-prefix
```
