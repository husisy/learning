# macOS

mac studio as Server

1. system upgrade, install `xcode` (app store)
2. enable ssh-server (system setting), VNC (system setting)
   * (optional, HKUST) connect with ethernet to have a public IP
   * ssh: connect mac via terminal
   * VNC: connnect mac with GUI
   * VNC (ipad user): install "Mocha VNC lite" (free for 5 minutes) or "Mocha VNC" (48 HKD)
3. install homebrew [link](https://brew.sh/)
   * `brew install git wget zsh tldr htop`
   * (optional) `zsh` bindkey [link](https://apple.stackexchange.com/a/114528)
4. install `conda`
   * macOS Apple M1 64-bit [link](https://docs.conda.io/en/latest/miniconda.html)
   * see code snippet below
5. when to use
   * ~~bilibili UP~~
   * pytorch
   * taichi

```bash
sw_vers
# ProductName:            macOS
# ProductVersion:         13.1
# BuildVersion:           22C65

# conda
conda create -y -n metal
conda install -y -n metal -c pytorch pytorch torchvision torchaudio
conda install -y -n metal -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum
```

```Python
import torch
import numpy as np
import multiprocessing
multiprocessing.cpu_count() #20
torch.backends.mps.is_available() #True
torch0 = torch.randn(2, 3, device='mps')
torch.sin(torch0)
```
