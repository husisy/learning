# linux learning

程序员会学习很多技术工具：操作系统linux、bash、Python。由于历史发展的原因，这些工具的功能可能会有交叠，例如bash和python都可以做计算器。对于重复的实现，存在更擅长执行该任务的工具，毋庸置疑相比bash，Python更适合作为通用计算器，bash计算器的表达能力较弱（纯文本，无类型系统），而同时Python并没有因为更强的表达能力而导致使用起来繁琐（我想大家还是会比较同意Python的学习曲线比较平缓）。程序员需要为每一个任务找到合适的工具（计算器使用Python），而将bash从计算器中释放出来（减少依赖），这样bash能够想更适合它的场景发展。

`linux/linux/README.md`与`linux/shell/README.md`耦合较多

1. link
   * [ubuntu documentation](https://help.ubuntu.com/)
   * [centos documentation](https://wiki.centos.org/Documentation)
   * [CentOS gitbook](https://www.gitbook.com/book/ninghao/centos/details)

## install ubuntu desktop system

1. link
   * [create a bootable USB on windows](https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-windows#0)
   * [install ubuntu desktop](https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop#0)
   * [install ubuntu server](https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-server#0)
   * [Ask ubuntu](https://askubuntu.com/)
   * [ubuntu forums](https://ubuntuforums.org/)
   * [知乎-Intel和AMD与x86, arm, mips有什么区别？](https://www.zhihu.com/question/63627218)
2. download
   * [Rufus](https://rufus.ie/)
   * [ubuntu](https://ubuntu.com/download/desktop)
3. select `FreeDOS`
4. click on `SELECT` (on the right of the `FreeDOS`)
5. select `MBR`
6. select `BIOS (or UEFI-CSM)`
7. possible warning
   * `Download required`, select `yes`
   * `ISOHybrid image detected`, select `yes`
   * `Rufus`, select `OK`
8. UEFI settup
   * surface: hold `volume-up`, press and release `power`
   * other PC: maybe hold `F12`
9. select `install Ubuntu`
    * or select `Try Ubuntu` (can also install Ubuntu from this mode)
10. recommended
    * select `Normal installation`
    * select `Download updates while installing Ubuntu`
    * select `install third-party software for xxx`
11. allocate drive space

## maintain ubuntu OS

1. 专有名词
   * LTS: long-term support
2. shell program: `sh`, `bash`
3. install software: `apt`, `apt-add-repository`, `apt-get`
4. shortcut
   * `Ctrl + Alt + F1`: desktop
   * `Ctrl + Alt + F2`: tty2 console
   * `ctrl+alt+T`: terminal
5. 偏见
   * **禁止**在文件名中包含任何标点符号punctuation `* ?`
   * 文件名**仅**可以使用letter`A-Za-z` + number`0-9` + underscore`_` + hyphen`-`

## filesystem

1. link
   * [gitbook / 鸟哥的 Linux 私房菜](https://legacy.gitbook.com/book/wizardforcel/vbird-linux-basic-4e/details) [gitbook](https://wizardforcel.gitbooks.io/vbird-linux-basic-4e/content/index.html)
   * [github-linuxtools](https://github.com/me115/linuxtools_rst)
2. 文件权限
   * `rwxs`：读、写、执行、
   * `d`：目录
   * `-`：普通文件
   * `l`：链接文件
   * `b`：设备文件里面的可供储存的周边设备（可随机存取设备）
   * `c`：设备文件里面的序列埠设备，例如键盘、鼠标（一次性读取设备）
   * `s`：数据接口文件
   * `p`：FIFO pipe
3. 修改文件权限：`chgrp chown chmod`
4. 用户列表：`/etc/passwd`
5. group列表：`/etc/group`
6. Filesystem Hierarchy standard (FHS)
   * 可分享的：可以分享给其他系统挂载使用的目录，包括可执行文件与使用者的邮件等数据
   * 不可分享的：自己机器上面运行的设备文件或者是与程序有关的socket文件等，仅与自身机器有关
   * 不变的：不会经常变动的，跟随着distribution而不变动。例如函数库、文件说明文档、系统管理员所管理的主机服务配置文件等等
   * 可变动的：经常改变的数据，例如登录文件、一般用户可自行收受的新闻群组等
   * `/`：与开机系统相关
   * `/usr` (unix software resource)：与软件安装/执行相关
   * `/var` (variable)：与系统运行过程相关

## disk management

1. link
   * [ubuntu documentation / iinstall a new hard drive](https://help.ubuntu.com/community/InstallingANewHardDrive)
   * [ask ubuntu / how do i add an additional hard drive](https://askubuntu.com/q/125257)
2. tools `gparted parted fdisk`
3. modify reserved space `sudo tune2fs -m  1 /dev/sda`
4. `sudo parted -l`
5. disk driver
   * `/dev/md` Multiple Device driver aka Linux Software RAID, [wiki/mdadm](https://en.wikipedia.org/wiki/Mdadm)
   * Redundant Array of Inexpensive Disks (RAID)

```bash
sudo parted /dev/sda
(parted) mklabel gpt
(parted) unit TB
(parted) mkpart
Partition name?  []? primary
File system type?  [ext2]? ext4
Start? 0
End? 4
(parted) print
```

```bash
# https://stackoverflow.com/q/139261
dd if=/dev/zero of=/dev/shm/test00 bs=1M count=128
```

## memory management

1. drop cache [stackexchange](https://unix.stackexchange.com/q/17936)
