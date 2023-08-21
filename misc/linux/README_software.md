# linux software

alphabetic order

TODO

1. [ ] scp
2. [ ] sftp
3. [ ] fz
4. [ ] sz

## misc

```bash
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@hostname\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\$ '

omz update #update oh-my-zsh
```

## apt package manager

1. link
   * [ubuntu - package management](https://help.ubuntu.com/lts/serverguide/package-management.html)
   * [apt user guide](https://www.debian.org/doc/manuals/apt-guide/index.en.html)
2. `dpkg`: debian-based package manager, can install, remove, build packages, but cannot automatically download and install packages or their dependencies
   * 已安装package: `dpkg -l`
   * package `ufw`安装的文件：`dpkg -L ufw`
   * 文件对应的package: `dpkg -S /etc/host.conf`，安装过程中产生的文件未必被包管理器记录
   * 本地安装：`sudo dpkg -i zip_3.0-4_i386.deb`
   * 卸载package（不解决依赖关系，不推荐）：`sudo dpkg -r zip`
3. `apt`: advanced packaging tool, install, upgrade existing packages, update package index, upgrade the ubuntu system
   * 帮助：`apt help`
   * 安装：`sudo apt install nmap`
   * 卸载：`sudo apt remove nmap`
   * 彻底卸载（清除配置文件）：`sudo apt remove --purge nmap` （TODO，配置文件是啥，apt怎么知道啥是配置文件）
   * update package index: `sudo apt update`
   * upgrade package (update index first): `sudo apt upgrade`
   * apt repository: `/etc/apt/sources.list`, `/etc/apt/sources.list.d`
   * 安装卸载记录：`/var/log/dkpg.log`
   * `sudo apt autoremove`
   * `apt list --upgradeable`
   * `apt list --installed`
4. `sudo aptitude`
5. automatic update
   * `sudo apt install unattended-upgrades`
   * `/etc/apt/apt.conf.d/50unattended-upgrades`
   * `/etc/apt/apt.conf.d/20auto-upgrades`
   * `/etc/cron.daily/apt`
   * notification see `sudo apt install apticron`, `/etc/apticron/apticron.conf`
6. misc
   * 关闭自动更新 `sudo dpkg-reconfigure -plow unattended-upgrades` [stackoverflow](https://unix.stackexchange.com/a/470710)

## choco package manager

[choco-package-list](https://chocolatey.org/packages)

```bash
choco --help
choco search
choco info
choco install
choco upgrade
chco uninstall
```

## ClamAV

1. link
   * [official site](https://www.clamav.net/)
2. `--exclude-dir="^/sys"`: The files in `/sys` are not real files, viruses will not infect them; [ClamAV: Can't read file ERROR](https://askubuntu.com/questions/591964/clamav-cant-read-file-error)
3. `clamscan --database=/root/Downloads/clamav_database20190104 -r --exclude-dir="^/sys" --log=./SCAN_LOG_09012019 /`

## curl

```bash
curl www.baidu.com
curl -o index.html www.baidu.com
curl -O https://raw.githubusercontent.com/airbnb/javascript/master/README.md #save as README.md
```

## diff

1. link
   * [GNU-documentation](https://www.gnu.org/software/diffutils/manual/html_node/Unified-Format.html)
   * [阮一峰/读懂diff](http://www.ruanyifeng.com/blog/2012/08/how_to_read_diff.html)
2. 正常格式normal diff，上下文格式context diff，合并格式unified diff

## driver

1. link
   * cuda toolkit安装流程见官方文档 [link](https://developer.nvidia.com/cuda-downloads)
   * cudnn安装流程 [link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux)
2. `ubuntu-drivers devices`, see [link](https://zhuanlan.zhihu.com/p/59618999)
3. `sudo ubuntu-drivers autoinstall`
4. `sudo killall -SIGQUIT gnome-shell`, see [link](https://askubuntu.com/a/100228)
5. cuda driver compatibility see [link](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
   * `cuda11.1`: `gpu>=455`
   * `cuda11`: `gpu>=450`
   * `cuda10.2`: `gpu>=440`
   * `cuda10.1`: `gpu>=418`
   * `cuda10.0`: `gpu>=410`
6. 使用conda安装pytorch：只需要系统安装好版本适配的GPU驱动，不需要在系统安装CUDA toolkit
   * conda负责分发合适的cuda toolkit，pytorch亦是基于该库进行编译
   * conda分发的cuda toolkit只是nvidia官网cuda toolkit的子集，至少不包含`nvcc`二进制文件
   * see [anaconda documentation](https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/)
7. nvidia dgx systems: [documentation / update](https://docs.nvidia.com/dgx/dgx-station-user-guide/index.html#upgrading-dgx-station-software)

```bash
# install operation see official documentation, should be something like below
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# https://github.com/dmlc/cxxnet/issues/108
sudo nvidia-smi -pm 1
```

cudnn

```bash
# TODO
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.2/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64
sudo chmod a+r /usr/local/cuda-11.2/include/cudnn*.h /usr/local/cuda-11.2/lib64/libcudnn*
```

## editor

1. `nano`
   * 保存：`ctrl + o`
   * 退出：`ctrl + x`
2. `vi`: `i`, `esc`, `/`, `:wq`, `:q!`
3. `emacs`
4. `joe`

## firewall

centos

```bash
sudo systemctl start firewalld
sudo systemctl stop firewalld
sudo systemctl status firewalld
```

ubuntu

```bash
sudo service ufw status
```

## firmware

```bash
# https://askubuntu.com/q/1153546
sudo service fwupd start
sudo fwupdmgr refresh #proxy maybe is required
sudo fwupdmgr update
```

## imgcat

1. link
   * [github](https://github.com/danielgatis/imgcat)

```bash
# macos
brew install danielgatis/imgcat/imgcat
```

## iostat

1. link
   * [wiki/iostat](https://en.wikipedia.org/wiki/Iostat)
   * [openbsd/iostat](https://man.openbsd.org/iostat)
   * [易百教程/iostat](https://www.yiibai.com/linux/iostat.html)
2. 相关 `mpstat netstat sar systat vmstat`
3. install `apt install sysstat`
4. 常用命令
   * `iostat`
   * `iostat --human -d sda -c`

## iperf3

1. link
   * [official-site](https://iperf.fr/)
   * [documentation](https://iperf.fr/iperf-doc.php)
2. install `apt install iperf3`
3. 使用
   * server：`iperf3 -s --port 9710`
   * client：`iperf3 -c 47.101.176.10 --port 9710`

## lsof

1. link
   * [linux-tools-quick-tutorial/lsof](https://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/lsof.html)

```bash
lsof | less
lsof -i :9792
lsof -c ssh
lsof -u xxx-username
lsof -p xxx-pid
```

## nfs

1. link
   * [ubuntu-server-doc/Network-File-System](https://ubuntu.com/server/docs/service-nfs)
2. install `apt install nfs-kernel-server nfs-common`
3. startup `service nfs-kernel-server start`
4. 支持指定端口，但不支持ssh端口转发`mount.nfs: requested NFS version or transport protocol is not supported`

```bash
# server
sudo nano /etc/exports
# /opt/tbd00  *(rw,sync,no_subtree_check)
sudo exportfs -a

# client
sudo mount 127.0.0.1:/opt/tbd00 /opt/tbd01
sudo mount -t nfs 127.0.0.1:/opt/tbd00 /opt/tbd01
sudo umount 127.0.0.1:/opt/tbd00
sudo mount -o port=2049 -t nfs 127.0.0.1:/opt/tbd00 /opt/tbd01
```

## nmcli

1. link
   * [documentation](https://developer-old.gnome.org/NetworkManager/stable/nmcli.html)
   * [askubuntu-link](https://askubuntu.com/q/461825) How to connect to WiFi from the command line

```bash
nmcli --help
nmcli connection #nmcli c
nmcli device #nmcli d
sudo nmcli d wifi rescan
nmcli d wifi list
sudo iwlist wlp4s0 scanning
nmcli c delete id xxx
```

## polipo

install polipo on ubuntu-20.04 [link](https://gist.github.com/fonsecas72/0ef04265a3d0c5822e5d441b8d2de1f8)

```bash
wget http://archive.ubuntu.com/ubuntu/pool/universe/p/polipo/polipo_1.1.1-8_amd64.deb
sudo dpkg -i polipo_1.1.1-8_amd64.deb

polipo -c ./polipo.config
nohup polipo -c ./polipo.config > ~/software/polipo.log 2>&1 &
```

```ini
socksParentProxy = localhost:1080
diskCacheRoot=""
proxyAddress = "0.0.0.0"
proxyPort = 8123
```

## ripgrep

1. link
   * [github](https://github.com/BurntSushi/ripgrep)
   * [user-guide](https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md)
   * [regex-supported](https://docs.rs/regex/1.5.4/regex/#syntax)
2. 支持windows/linux/macos
3. install `apt install ripgrep`
4. automatic filtering (default)
   * `.gitignore`文件中记录的 `--no-ignore`
   * 隐藏文件以及隐藏文件夹 `--hidden`
   * 二进制文件
   * 符号链接 `--follow`, `-L`
   * `--debug`
5. 特殊文件`.gitignore`, `.ignore`, `.rgignore`
6. ripgrep**永远不会**在位修改文件，ripgrep仅修改打印输出内容

## screen

1. link
   * [official site](https://savannah.gnu.org/projects/screen)
   * [GUN-official-site](https://www.gnu.org/software/screen/)
   * [documentation](https://www.gnu.org/software/screen/manual/html_node/index.html)
   * [linux 技巧：使用 screen 管理你的远程会话](https://www.ibm.com/developerworks/cn/linux/l-cn-screen/index.html)
   * [linux screen的用法](https://www.jianshu.com/p/e91746ef4058)
   * [StackExchange-how to split the terminal in to more than one view](https://unix.stackexchange.com/a/7455)
2. similar package: tmux, tmate
3. install
   * `sudo yum install screen`
   * `sudo apt-get intall screen`
4. 进入screen前**务必**`conda deactivate`
5. 基础命令
   * create: `screen -S xxx`
   * detach: `ctrl+a` -> `d`
   * show: `screen -list`, `screen -ls`
   * restore: `screen -r xxx`
   * kill: `ctrl+a` -> `k`
   * help: `ctrl+a` -> `?`
6. 进程组process group：进程的集合；唯一进程组ID
7. 会话期session：进程组的集合；会话期首进程 session leader；会话期ID为首进程的ID
8. 控制终端controlling terminal
9. 控制进程controlling process：与控制终端连接的会话期首进程
   * 前台进程组：当前与终端交互的进程
   * 后台进程组
10. 多个进程之间多路复用一个物理终端的全屏窗口管理器

## sed

1. link
   * [gnu/documentation](https://www.gnu.org/software/sed/manual/sed.html)

```bash
sed 's/hello/world' input.txt > output.txt
sed 's/hello/world' < input.txt > output.txt
cat input.txt | sed 's/hello/world' - > output.txt

sed -n '269p' README_software.md
sed -n '269p;270p' README_software.md
```

## ssh

1. link
   * [ubuntu doc - OpenSSH Server](https://help.ubuntu.com/lts/serverguide/openssh-server.html)
   * [阮一峰 - SSH原理与运用一](http://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html)
   * [阮一峰 - SSH原理与运用二](http://www.ruanyifeng.com/blog/2011/12/ssh_port_forwarding.html)
   * [ssh-jump-host](https://wiki.gentoo.org/wiki/SSH_jump_host)
2. 安装
   * server（ubuntu desktop默认不安装）: `sudo apt install openssh-client openssh-server`
   * windows client: **TODO**
3. ssh基本用法：`ssh -p 23333 username@233.233.233.233`
4. ssh密码登录过程
   * 远程主机收到用户的登录请求后，把公钥发给用户
   * 用户使用公钥将登录密码加密后，将加密后内容发给远程主机
   * 远程主机使用私钥解密登录密码，如果密码正确，即同意用户登录
5. 中间人攻击
6. ssh公钥登录过程
   * 公钥位于远程主机，私钥位于本地
   * 远程主机收到用户的登录请求后，把一段随机字符串发给用户
   * 用户用私钥加密字符串后，发给远程主机
   * 远程主机用公钥进行解密，如果成功，就证明用户是可信的，直接允许登录shell
7. 配置文件：`/etc/ssh/sshd_config`
   * **备份**：`sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.original`
   * **备份**：`sudo chmod a-w /etc/ssh/sshd_config.original`
   * 端口：`Port 2333`
   * 公钥私钥登录：`PubkeyAuthentication yes`
8. 公钥私钥登录
   * `ssh-keygen -t rsa`
   * 生成的公钥：`~/.ssh/id_rsa.pub`
   * 私钥：`~/.ssh/id_rsa`
   * 添加公钥至remote host `~/.ssh/authorized_keys`: `ssh-copy-id username@remotehost`
   * 文件夹`~/.ssh`权限`700`，文件`~/.ssh/authorized_keys`权限`600`
9. 重启ssh服务
   * `sudo service ssh restart`
   * `sudo /etc/init.d/ssh restart`
   * `sudo systemctl restart sshd.service`（ubuntu文档中提及，但在WSL中失败）
10. 绑定本地端口，**TODO SSH原理与运用（二）：远程操作与端口转发**
11. 本地端口转发
    * host-pc通过localhost链接host-server的jupyter服务：`ssh -L 127.0.0.1:18888:127.0.0.1:8888 username@host-server`
    * 后台运行: `nohup autossh -NT -L 127.0.0.1:18888:127.0.0.1:8888 username@host1 > ~/software/autossh.log 2>&1 &`
    * 然后host-pc即可访问`127.0.0.1:18888`
    * `localhost`与`127.0.0.1`为同义词
12. 远程端口转发
    * host0通过将本地ssh端口转发到host1: `ssh -R 0.0.0.0:2022:127.0.0.1:22 username@host1`
    * 后台运行: `nohup autossh -NT -R 0.0.0.0:2022:127.0.0.1:22 username@host1 > ~/software/autossh.log 2>&1 &`
    * host-pc通过host1端口登录到host0: `ssh -p 2022 username@host1`
13. `telnet`, `rcp`不安全
14. host-win通过ssh连接host-server，在host-server上执行绘图代码，绘图窗口在host-win打开
    * host-win运行xming（此时host-win的WSL图形界面可运行）
    * powershell中设置环境变量 `$env:DISPLAY="localhost:0.0"`，参考xming设置，默认`localhost:0.0`
    * host-win发起ssh连接 `ssh -Y username@host-server`
    * 在ssh-shell中执行命令（预期在本地打开绘图窗口） `python -c "import matplotlib.pyplot as plt; plt.plot([0,1],[0,1]); plt.show()"`
    * 注：测试发现WSL/linux平台使用参数`-X`即可，但win-powershell使用`-X`参数失败
    * 注：mobaxterm软件默认支持x11 forwarding，故不需要上述配置
15. 限制root-ssh登录
    * What does 'without password' mean in sshd_config file?, [askubuntu](https://askubuntu.com/questions/449364/what-does-without-password-mean-in-sshd-config-file)
    * HOW DO I DISABLE SSH LOGIN FOR THE ROOT USER? [link](https://mediatemple.net/community/products/dv/204643810/how-do-i-disable-ssh-login-for-the-root-user)
16. hydra: max ssh login attempts, [link](https://serverfault.com/questions/275669/ssh-sshd-how-do-i-set-max-login-attempts)
17. 一个terminal配置多个github账号 [link](https://gist.github.com/rbialek/1012262) [link](https://www.freecodecamp.org/news/manage-multiple-github-accounts-the-ssh-way-2dadc30ccaca/)
18. misc
    * `ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=1 <host>`, 保存ssh连接较长时间存活 [link](https://unix.stackexchange.com/a/34201)
    * `ssh-keygen -t ed25519 -a 100` [stackexchange-link](https://security.stackexchange.com/a/144044)

```bash
proxy --port 23333 --hostname 127.0.0.1
ssh -L 127.0.0.1:23333:127.0.0.1:23333 -o ServerAliveInterval=5 -o ServerAliveCountMax=1 test01@123.456.78.9 -i proxy.id_rsa
export http_proxy=http://127.0.0.1:23333

hydra -l tbd01 -P download/thc-hydra-8.8/dpl4hydra_full.csv -t 4 -s 2480 ssh://123.456.78.9
iptables -A OUTPUT -p tcp -d 987.654.32.1 -j REJECT
```

### ssh port forwarding

分为local forwarding和remote forwarding，分别对应`ssh -L xxx`和`ssh -R xxx`

服务器/本地机器/host

使用场景一：远端机器`host-remote`运行jupyter，本地机器`host-local`使用浏览器来打开jupyter页面

说明

1. 具体场景：在`dgx`超算上运行`jupyter`，在个人笔记本电脑上来访问其网页。这里超算的身份便是`server`，个人笔记本电脑便是`client`
2. 此处的jupyter可以替换为ssh-server, web-server, proxy-server等
3. `server`和`client`是指某一网络通信过程中的主客身份，场景一中涉及的网络通信有「端口转发」和「ssh登录」
   * 与机器是台式机/笔记本无关，与机器操作系统是win/linux/mac无关
   * 端口转发过程中的`server`是`host-remote`，`client`是`host-local`
   * ssh过程中的`server`是`host-remote`，`client`是`host-local`，正好与端口转发一致
   * 由于端口转发会使得`server/client`较为复杂，一个机器可能同时是client也是server，在后续场景中的`server/cliient`请自行分析

操作

1. 启动jupyter：在`host-remote`上执行`jupyter lab`
   * 默认端口是`8888`
2. `host-local`通过ssh登录`host-remote`：在`host-local`上执行`ssh -L 127.0.0.1:23333:127.0.0.1:8888 username@ip`
   * 将`username`替换为真实的用户名，将`ip`替换为真实的ip
   * `127.0.0.1:23333`是`host-local`的端口
   * `127.0.0.1:8888`是`host-remote`的端口，即步骤一中启动jupyter占用的端口，如步骤一中修改为其他端口，这里也需做相应修改
   * `-L`表示local forwarding
3. 浏览器访问jupyter：在`host-local`浏览器中输入网址`127.0.0.1:23333`即可
   * 数据传输途径：浏览器 --> `127.0.0.1:23333`(local) --ssh--> `127.0.0.1:8888`(remote) --> jupyter

使用场景二：本地机器`host-local`运行jupyter，远端机器`host-remote`浏览器访问jupyter页面

说明

1. 具体场景
   * 个人电脑上启动了jupyter，想要让同事（同事在host-remote）来访问jupyter页面
   * 个人电脑上启动了proxy-server，想要让超算服务器访问本地电脑的proxy-server
2. 此处端口转发的`server/client`与ssh的`server/client`相反

操作

1. 启动jupyter：在`host-local`上执行`jupyter lab`
   * 默认端口是`8888`
2. `host-local`通过ssh登录`host-remote`：在`host-local`上执行`ssh -R 127.0.0.1:23333:127.0.0.1:8888 username@server-ip`
   * 将`username`替换为真实的用户名，将`server-ip`替换为真实的ip
   * `127.0.0.1:23333`是`host-remote`的端口
   * `127.0.0.1:8888`是`host-local`的端口，即步骤一中启动jupyter占用的端口，如步骤一中修改为其他端口，这里也需做相应修改
   * `-L`表示remote forwarding
3. 浏览器访问jupyter：在`host-remote`浏览器中输入网址`127.0.0.1:23333`即可
   * ssh数据途径：`host-local` --> `host-remote`
   * jupyter数据传输途径：浏览器 --> `127.0.0.1:23333`(remote) --ssh--> `127.0.0.1:8888`(local) --> jupyter

使用场景三：`host0`运行了jupyter，`host1`是本地电脑，`host2`想要访问jupyter页面

1. 在`host1`上执行`ssh -R 127.0.0.1:23333:host0-ip:8888 username@host2-ip`
   * 将`host0-ip`替换为真实ip，将`username`替换为真实用户名，将`host2-ip`替换为真实ip
2. 在`host2`上访问`127.0.0.1:23333`

## systemctl

```bash
sudo systemctl start x
sudo systemctl stop x
sudo systemctl status x
sudo systemctl restart x
sudo systemctl reload x
sudo systemctl enable x #auto start after restart computer
sudo systemctl disable x
```

## tar

```bash
tar -xzf xxx.tar.gz
tar -xf xxx.tar.xz
tar -cf airbnb-js.tar README.md
tar -xf airbnb-js.tar
tar -czf airbnb-js.tar.gz README.md
tar -xzf airbnb-js.tar.gz
```

## tldr

a collection of community-maintained help pages for command-line tools, that aims to be a simpler, more approachable complement to traditional man pages

1. link
   * [github](https://github.com/tldr-pages/tldr)
2. install `npm install -g tldr`
3. quickstart
   * `tldr tldr`
   * `tldr git`
   * `tldr git pull`
   * `tldr tar`

## tmate

1. link
   * [official site](https://tmate.io/)
   * [github](https://github.com/tmate-io/tmate)

## tmux

1. link
   * [github](https://github.com/tmux/tmux)
   * [github-wiki](https://github.com/tmux/tmux/wiki)
   * [阮一峰/Tmux使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)
2. install `apt install tmux`
3. get help `man 1 tmux`
4. concept: tmux server, tmux client
5. 按键说明
   * `C-b`: press `Ctrl + b`
   * `M-b`: press `Alt + b` (meta)
   * `S-b`: press `Shift + b`
   * `C-b c`: press `Ctrl + b`, release both keys, then press `c`
   * `C+b C-c`: press `Ctrl + b`, release both keys, then press `Ctrl + c`
   * `C-b C-b`: send `C-b` to the program in the active pane
6. 常用命令
   * `tmux list-keys`, `tmux lsk`, `C-b ?` view mode of key binding, `q` to quit
   * `tmux new-window`, `tmux neww`, `C-b c`
   * `tmux select-window -t 0`
   * `tmux new-session`, `tmux new`, `tmux`
   * `tmux detach` `C-b d`
   * `tmux attach -t 0`
   * `tmux list-sessions`, `tmux ls`, `C-b s`
   * `tmux list-commands`
   * `tmux kill-session -t 0`
   * `tmux switch -t 0`
   * `tmux rename-session -t 0 XXX` `C-b $`
   * `tmux split-window`, `C-b "`
   * `tmux split-window -h`, `C-b %`
   * `tmux select-pane -U`, `-U/D/L/R`, `C-b up-arrow`, `C-b down-arrow`
   * `tmux swap-pane -U`, `-U/D/L/R`

ws00（同一行的多个命令表示等价命令）

1. `tmux` 启动tmux-session（同时启动一个tmux-window）
2. `exit` 关闭tmux-session（同时关闭最后一个tmux-window）

ws01

1. `tmux` 启动tmux-session
2. `tmux new-windows`, `tmux neww` 再启动一个tmux-window，此时有俩tmux-windows
3. `exit` 关闭第二个tmux-windows
4. `exit` 关闭第一个tmux-windows

ws02

1. `tmux` 启动tmux-session
2. `tmux detach` 暂时退出tmux-session，此时tmux-session未关闭
3. `tmux list-sessions`, `tmux ls` 查看所有tmux-session
4. `tmux attact -t 0` 连接session-name为`0`的tmux-session
5. `exit` 关闭tmux-session

## update-alternatives

```bash
# see https://askubuntu.com/a/26518
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 20

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 20

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
sudo update-alternatives --set cc /usr/bin/gcc

sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
sudo update-alternatives --set c++ /usr/bin/g++
```

## yum

1. `/etc/yum.repos.d/`

```bash
yum repolist

# IUS repo
sudo yum install https://centos7.iuscommunity.org/ius-release.rpm

# mariadb
yum search mariadb
yum info mariadb
sudo yum install mariadb101u
sudo yum remove mariadb-libs
```
