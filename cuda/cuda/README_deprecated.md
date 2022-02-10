# cuda installation

**deprecated** 保留仅用于必要时查询

1. link
   * [official site](https://developer.nvidia.com/computeworks)
   * [cuda quick start guide]( https://developer.download.nvidia.com/compute/cuda/9.1/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf)
   * [Windows Installation Guide]( http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
   * [Mac Installation Guide]( http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/)
   * [Linux Installation Guide]( http://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

使用说明（普通用户）

1. 向`~/.bashrc`文件中添加如下命令，以`cuda-10.0`为例
   * `export PATH="/path/to/cuda-10.0/bin:$PATH"`
   * `export LD_LIBRARY_PATH="/path/to/cuda-10.0/lib64:/path/to/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH"`
   * `export CUDA_VISIBLE_DEVICES=1`：建议添加该变量，以避免默认占用所有GPU，如的确需要多GPU，建议在程序代码中修改该环境变量
   * 重新ssh登录后生效
2. 检验是否配置成功
   * `nvcc --verion`显示CUDA compiler的版本信息
   * `nvidia-smi`显示GPU的运行状态（不做上述配置也能执行该命令）
   * `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"`测试tensorflow2.0
   * `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`测试tensorflow2.1
3. pip安装的tensorflow需要特定版本的cuda与cudnn，见[tensorflow-release](https://github.com/tensorflow/tensorflow/releases)，如cuda版本不匹配，则需要自行编译安装tensorflow

前期检查（服务器管理员）

1. CUDA-Capable GPU
   * `lspci | grep -i nvidia`
   * [nvidia-list]( https://developer.nvidia.com/cuda-gpus)
2. supported linux:  `uname -m && cat /etc/*release`
3. gcc installation check: `gcc --version`
4. ubuntu建议使用apt安装nvidia driver，使用runfile安装cuda
5. kernel headers and development packages installation check
   * `uname -r`
   * RHEL/CentOS: `sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)`
   * Fedora
   * OpenSUSE/SLES
   * Ubuntu
   * `uname -m && cat /etc/*release`
6. uninstall if necessary
   * `/path/to/cuda-X.Y/bin/uninstall_cuda_X.Y.pl`
   * `/path/to/nvidia-uninstall`
   * Redhat/CentOS: `sudo yum remove <package_name>`
   * Fedora: `sudo dnf remove <package_name>`
   * OpenSUSE/SLES: `sudo zypper remove <package_name>`
   * Ubuntu: `sudo apt-get --purge remove <package_name>`

RPM Installer（服务器管理员）

虽然该安装方式为*官方推荐*，但本人之前使用该方式未能成功安装多个版本CUDA（应该时可以做的，cuda-soft-link啥的），后人如测试成功，望告知

Runfile Installer（服务器管理员）

1. 下载必要文件cuda
   * 下载链接可由nvidia官网处获取，或者下载至PC后ftp上传至服务器
   * 历史版本的cuda见legacy
   * `wget #cuda-10.0`，以下用`cuda_xxx.linux.run`作为其文件名
2. 安装
   * 「CUDA安装」与「GPU driver安装」是独立的，即可以「只安装cuda不安装GPU driver」，也可以「只安装GPU driver不安装cuda」，但需要两者都安装，方能运行cuda
   * 「CUDA安装」和一般的软件一样，**不需要**重启服务器，**不需要**设置runlevel，**不需要**管理员权限
   * 「GPU driver安装」可能需要重启服务器等一系列复杂的配置（未测试）
   * `bash /path/to/cuda_xxx.linux.run`，安装CUDA主文件：自行判断是否安装GPU driver，建议安装sample
3. 下载cudnn
   * 下载链接可由nvidia官网处获取，或者下载至PC后ftp上传至服务器
   * `wget #cudnn`，以下用`cudnn-xxx.tgz`作为其文件名
4. 添加cudnn至CUDA安装路径
   * `tar -xzvf /path/to/cudnn-xxx.tgz`：解压cudnn
   * `sudo cp /path/to/cudnn/cuda/include/cudnn.h /path/to/cuda-10.0/include`
   * `sudo cp /path/to/cudnn/cuda/lib64/libcudnn* /path/to/cuda-10.0/lib64`
   * `sudo chmod a+r /path/to/cuda-10.0/include/cudnn.h`：添加其他用户读权限（似乎冗余步骤，测试发现默认情况下其他用户就有读权限）
   * `sudo chmod a+r /path/to/cuda-10.0/lib64/libcudnn*`
5. 卸载：`sudo /path/to/cuda-10.0/bin/uninstall_cuda_X.Y.pl`
6. 其它一些命令
   * `echo "$PATH" | tr ':' '\n'`
   * `cat /proc/driver/nvidia/version`
   * `sudo service lightdm stop`
   * `sudo init 3`
   * `lsmod | grep nouveau`
   * `sudo service lightdm start`
