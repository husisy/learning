# Python Environment

1. win-python层次关系
   * OS：无，通过python launcher for windows来管理USER-level
   * USER：在用户目录下可以安装任意个独立的python子版本，将一个常用版本路径及`Scripts`路径添加在`PATH`
   * PROJECT：通过`pipenv`生成独立的项目环境
2. linux-python层次关系
   * OS：系统单一环境，用于生成环境部署，普通用户无权限安装Python包
   * USER：在OS-level的基础上，普通用户通过`pip install --user`将Python包安装至个人目录
   * PROJECT：通过`pipenv`生成独立的项目环境
   * **仅可通过conda/pyenv解决多个python子版本问题**，但此时`pip install --user`的用户目录存在冲突问题
3. conda-python层次关系（几乎OS-independent）
   * OS：无
   * USER：存在默认环境`conda-base`，但不建议使用，win下仅在进入anaconda prompt时激活该默认环境，linux仅在`conda activate base`后激活该默认环境；在默认环境外，用户可以在个人目录下创建任意个独立环境
   * PROJECT：无
   * conda目前支持cmd与powershell
4. 包管理工具间的兼容性
   * `pip install conda`: anaconda官方抛出warning [link](https://pypi.org/project/conda/)
   * `conda install pip`：anaconda提供兼容支持，有不少包仅提供pip安装方式

综上，win/linux下

1. 软件开发：在用户目录下安装`anaconda/miniconda`来管理多个Python环境
2. 应用部署：不详

## quickstart

miniconda安装

1. 下载
   * windows：[下载链接](https://docs.conda.io/en/latest/miniconda.html)
   * linux: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
2. 安装
   * windows: 双击run run run
   * linux: `bash Miniconda3-latest-Linux-x86_64.sh`, run run run，安装完之后必须重新登录黑框框miniconda才会生效
3. win下是否应该将conda添加至全局路径PATH：**不建议**而且默认安装选项是不添加
   * 添加带来的便利：vscode可以“直接”在黑框框中运行代码
   * 添加至PATH会引发的异常包括但不限于：在其它环境下可以访问base的package与可执行文件（例如xxx-env未安装jupyter，base安装了jupyter，但用户可以在xxx-env启动jupyter，但启动的jupyter行为不正确）
   * 不添加至PATH但想使用vscode-run的解决方案：在powershell中手动执行如下命令（假设用户非常清楚该命令的含义）
   * `& '/PATH/TO/Miniconda3/shell/condabin/conda-hook.ps1'; conda activate '/PATH/TO/Miniconda3'`
4. optional 境内使用conda镜像
   * [清华大学镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
   * [上海交通大学镜像](https://mirrors.sjtug.sjtu.edu.cn/#/)
   * [北京外国语大学开源软件镜像站](https://mirrors.bfsu.edu.cn/)
5. 启动miniconda
   * windows：开始菜单的`anaconda prompt`应当作为conda环境的唯一入口
   * linux：登录黑框框即可
6. 安装环境（见下代码块）

```bash
# tensorflow-2.1 doesn't support python3.8 yet @20200422
# conda config --add channels conda-forge
conda create -n python_tf2
conda install -n python_tf2 -c conda-forge python=3.7 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum
conda activate python_tf2
pip install tensorflow
# conda-forge: opencv bokeh pydot nltk flask cchardet joblib scikit-learn seaborn scikit-image
```

for pytorch env

```bash
conda create -y -n nocuda
conda install -y -n nocuda -c pytorch pytorch torchvision torchaudio cpuonly
conda install -y -n nocuda -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum

conda create -y -n cuda112
conda install -y -n cuda112 -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -y -n cuda112 -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum
conda activate cuda112
pip install tensorflow
# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda create -y -n cuda113
conda install -y -n cuda113 -c conda-forge cudatoolkit=11.3
conda install -y -n cuda113 -c pytorch pytorch torchvision torchaudio
conda install -y -n cuda113 -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy nccl

conda create -y -n cuda117
# conda install -y -n cuda117 -c conda-forge cudatoolkit=11.7
conda install -y -n cuda117 -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.7
# conda install -y -n cuda117 -c pytorch pytorch torchvision torchaudio
conda install -y -n cuda117 -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy nccl

conda create -y -n cuda118
conda install -y -n cuda118 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -n cuda118 -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cupy nccl cvxpy scs

conda create -y -n metal
conda install -y -n metal -c pytorch pytorch torchvision torchaudio
conda install -y -n metal -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum
```

## conda

1. link
   * [github-conda](https://github.com/conda/conda)
   * [conda-documentation](https://docs.conda.io/en/latest/)
   * [conda-get-started](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
   * [Anaconda-documentation](https://docs.anaconda.com/anaconda-cloud/user-guide/)
   * [Anaconda多环境多版本python配置指导](https://www.jianshu.com/p/d2e15200ee9b)
   * [anaconda-cloud/getting-started](https://docs.anaconda.com/anacondaorg/user-guide/getting-started/#finding-downloading-and-installing-packages)
2. 支持语言: Python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN
3. 强烈建议使用个人目录的conda，而非个人目录的Python，而非公共目录的conda，而非公共目录的Python
   * 因各项目之间可能存在不兼容包依赖关系，故需要多个conda环境，故不建议使用个人目录或公共目录的Python
   * 因pip安装包会直接安装至Python目录下，进而污染整个Python目录，如出现问题则需要重装Python，而conda多个环境彼此不影响，可以方便的移除整个环境，因而不建议使用个人目录或公共目录的Python
   * 因公共目录的conda创建的环境存在用户冲突，用户A安装了`numpy`后存在的缓存文件会造成用户B无法正常安装`numpy`（解决方案很繁琐），因而不建议使用公共目录的conda
   * 产品部署：**不适用**
4. miniconda / anaconda: ananconda默认安装更多的包（例如Spyder），所以anaconda安装文件更大，但功能上两者完全等同，以下统称为conda
5. install all packages at one command, **NOT** one by one (may lead to dependency conflicts)
6. channel管理
7. 镜像源
8. pip与conda兼容问题：在每个conda环境中都有一个**独立的**pip包，故需要激活环境后方能使用对应环境下的pip包

conda基础命令

1. 常见参数缩写
   * `-e` / `--envs`
   * `-n` / `--name`
   * `-c` / `--channel`
2. 显示conda信息
   * `conda --version`
   * `conda info`
   * `conda info --envs`
3. 使用示例0
   * `conda create --name python_cpu` 创建环境`python_cpu`
   * `conda search --channel conda-forge numpy` 在`conda-forge`等channel搜索`numpy`，见[anaconda cloud](https://anaconda.org)
   * `conda install --name python_cpu --channel conda-forge numpy` 从`conda-force`向`python_cpu`环境安装`numpy`
   * `conda list --name python_cpu` 显示环境`python_cpu`下的所有安装包
   * `conda activate python_cpu`
   * `conda deactivate`
   * `conda remove --name python_cpu numpy` 卸载环境`python_cpu`中的`numpy`
   * `conda env remove --name python_cpu` 移除环境`python_cpu`
4. `conda update conda`
5. config
   * `conda config --set auto_activate_base false`
6. misc
   * `conda create -n python_cpu_dev --clone python_cpu`
   * `conda env export -n python_cpu > environment.yml`
   * `conda env create --file environment.yml`

## Pip Installs Packages (pip)

1. link
   * [documentation](https://pip.pypa.io/en/latest/reference/)
   * [Python Packaging User Guide](https://packaging.python.org/)
   * [Installing pip/setuptools/wheel with Linux Package Managers](https://packaging.python.org/guides/installing-using-linux-tools/)
   * why you should use `python -n pip` [link](https://snarky.ca/why-you-should-use-python-m-pip/)
2. install `pip`, `setuptools`, `wheel`
   * win: `python -m pip install --upgrade pip setuptools wheel`
   * ubuntu: `sudo apt install python3-venv python3-pip python3-setuptools python3-wheel`, `alias python=python3`, `alias pip=pip3`
   * more see [install pip/setuptools/wheel with linux package managers](https://packaging.python.org/guides/installing-using-linux-tools/)
3. update
   * win: `pip install --upgrade pip`
   * ubuntu OS-level: no necessary
   * ubuntu USER-level: no necessary, `pip install --user --upgrade pip`
4. user-base
   * linux: `python -m site --user-base`, `xxx/bin`
   * win: `python -m site --user-base`, `xxx/site-packages`->`xxx/Scripts`
5. install
   * `pip install xxx`
   * `pip install xxx=1.0.4`
   * `pip install xxx>=1.0.4`
   * `pip install -r requirements.txt`
   * `pip install --upgrade xxx`
   * `pip install xxx.whl`: install from local file
   * `pip install --user`: see [user installs](https://pip.pypa.io/en/latest/user_guide/#user-installs)
   * `subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xxx'])`
6. 常用命令
   * uninstall: `pip uninstall xxx`
   * list: `pip list`, `pip list --outdated`
   * show: `pip show xxx`, `pip show --files xxx`
   * search: `pip search xxx`
   * wheel: `pip wheel --wheel-dir DIR -r requirements.txt`
   * download: `pip download --destination-directory yyy xxx`, `pip download --destination-directory yyy -r requirements.txt`
   * `requiements.txt`: `pip freeze > requirements.txt`, see [link](https://pip.pypa.io/en/latest/reference/pip_install/#requirements-file-format)
7. 境内使用镜像
   * [清华大学镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
   * [tencent镜像](https://mirrors.cloud.tencent.com/help/pypi.html)

```bash
# win
pip install --upgrade pip setuptools wheel
pip install --upgrade cython ipython matplotlib pandas pylint jupyter Pillow scipy autopep8 tqdm pipenv jupyterlab h5py scikit-learn scikit-image protobuf graphviz lxml seaborn requests bokeh tensorflow

# linux
pip install --upgrade --user cython ipython matplotlib pandas pylint jupyter Pillow scipy autopep8 tqdm pipenv jupyterlab h5py scikit-learn scikit-image protobuf graphviz lxml seaborn requests bokeh tensorflow

# fail: opencv
```

## pipenv

1. link
   * [documentation](https://pipenv.readthedocs.io/en/latest/)
   * [github](https://github.com/pypa/pipenv)
2. install pipenv
   * win: `pip install pipenv`
   * linux: `pip install --user pipenv`
3. general workflow
   * `cd ws00`
   * `pipenv --three`
   * `pipenv install`
   * `pipenv install xxx`
   * `pipenv run python main00.py`
   * `pipenv shell`, run code in this new shell, then `exit`
4. `pipenv update --outdated`
5. specify Python
   * `pipenv --python 3`
   * `pipenv --python 3.6`
6. speicfy version
   * `pipenv install xxx>=1.4`
   * `pipenv install xxx>1.4`
   * `pipenv install xxx<=1.4`
   * `pipenv install xxx<1.4`
   * `pipenv install xxx~=1.4`
   * `pipenv install xxx!=1.4`

## venv

1. link
   * [documentation](https://docs.python.org/3/library/venv.html)
2. create virtual environment: `python -m venv ws00`
3. `pip`, `setuptools`, `wheel`
   * win (non linux package manager): `python -m pip install --upgrade pip setuptools wheel`
   * [ubuntu](https://packaging.python.org/guides/installing-using-linux-tools/#debian-ubuntu): `sudo apt install python3-venv python3-pip`
4. activate
   * bash/zsh: `source ws00/bin/activate`
   * fish: `. ws00/bin/activate.fish`
   * csh/tcsh: `source ws00/bin/activate.csh`
   * cmd: `ws00/Scripts/activate.bat`
   * PowerShell: `ws00/Scripts/Activate.ps1`
5. deactivate
   * bash/zsh: `source deactivate`
   * PowerShell: `deactivate`

## conda-forge

1. link
   * [official site](https://conda-forge.org/)
   * [github](https://github.com/conda-forge)
   * [documentation](https://conda-forge.org/docs/)
   * [documentation / becoming involved](https://conda-forge.org/docs/user/contributing.html#becoming-involved)
   * [github / miniforge](https://github.com/conda-forge/miniforge)

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

## conda mirror

1. link
   * [github/sjtug/mirror-docker](https://github.com/sjtug/mirror-docker)
   * [XUNGE-blog/搭建本地Anaconda镜像](https://xungejiang.com/2019/07/06/local-anaconda-mirror/)
   * [github/tunasync-scripts](https://github.com/tuna/tunasync-scripts)
   * [Dreambooker blog / set up personal anaconda mirror](https://dreambooker.site/2018/11/16/Set-up-personal-Anaconda-mirror/)
   * [tencent/手把手教你搭建Anaconda镜像源](https://cloud.tencent.com/developer/article/1618504)
