# macOS

本文档

1. 只记录winOS和macOS操作习惯上的对应关系
2. 不争论孰优孰劣
3. 不争论主观无法量化指标上的比较，例如优雅
4. 主要针对「键盘+触控板」的使用说明（笔者使用习惯），配合鼠标的使用习惯视情况增加
5. 中英文混合：笔者认知中较多的计算机术语用英文表述更容易被接受，例如键盘按键”Enter/return”（回车）, “ctrl”（控制）, “shift”（换档）对应的中文表述都不是太常见

术语约定（笔者不清楚其官方名称，如有错望指出）

1. macOS
   * 桌面desktop：三指上滑最上方那个几个块块，每个块块视为一个桌面
   * 窗口：三指上滑出现的块块，每个块块视为一个窗口
   * 应用：chrome，vscode等，一个应用可以有多个窗口
   * menu bar: 屏幕右上角区域，通常包含battery, wifi等
   * notification center: 屏幕最右上角区域，点击触发widget（双指从右侧划入，或者`fn+n`）
   * macOS-shortcut [link](https://support.apple.com/en-hk/HT201236)
2. 特殊按键命名约定
   * winOS: `ctrl`, `fn` (function), `shift`, `alt`, `super` (see [wiki/super-key](https://en.wikipedia.org/wiki/Super_key_(keyboard_button)))
   * macOS: `ctrl`, `fn/globe` (function), `shift`, `option`, `command`
3. 按键记号说明
   * 加号表示同时按下，`ctrl+c`
   * 逗号表示依次按下，`ctrl+k,z` (vscode zen mode in winOS)
   * `arrow`代指上下左右方向键
   * `backtick` 反引号，一般是键盘中数字`1`左边那个键，鉴于markdown中该符号有特殊含义，故本文用`backtick`替代
4. 笔者偏见
   * 使用搜狗输入法替代系统拼音输入法
   * macOS reduce motion: system setting, accessibility

命令行获得macOS状态

```bash
sw_vers
# ProductName:            macOS
# ProductVersion:         13.1
# BuildVersion:           22C65

# architecture
uname -a #arm64

# ip address
ipconfig getiflist #list all interface
ipconfig getifaddr en0 #en1

# get cpu info
sysctl -a | grep cpu

# detect m1 or m2
sysctl -n machdep.cpu.brand_string #Apple M1 Ultra, Apple M2
```

操作对应关系

1. search everything
   * winOS: `super`
   * macOS, spotlight search: `command+space`
2. 中英文输入法切换
   * winOS: `super+space`, `shift`
   * macOS: `fn`, `ctrl+space`
3. 特殊字符输入
   * winOS: `ctrl+shift+b`
   * macOS: `option+shift+b`
   * 搜狗输入法：`v,1`
4. 切换
   * winOS切换窗口: 三指左右滑动，四指上划然后选择, `ctrl+tab`, `super+tab`然后选择
   * winOS切换desktop: 四指左右滑动切换desktop
   * macOS: `command+tab` 不同应用间切换
   * macOS: `command+backtick` 同一应用间不同窗口切换
   * macOS: `ctrl+tab` 当前窗口的tab之间切换，例如chrome，finder，vscode，terminal
   * macOS: `command+1/2/3` 跳转至当前窗口的第x个tab
   * macOS: 三指上划然后选择窗口
   * macOS: 四指左右滑动切换desktop，`ctrl+arrow`
5. split window
   * winOS: `win+arrow`
   * macOS: TODO
6. new
   * winOS: `ctrl+n`, `ctrl+t` TODO
   * macOS: `command+shift+n` chrome创建新匿名窗口，vscode创建新窗口
   * macOS: `command+n` 同一应用创建新窗口，例如chrome，finder，terminal。vscode创建新tab
   * macOS: `command+t` 当前窗口创建新tab，例如chrome，finder，terminal
7. 文本编辑跳转go to，例如vscode，word等场景，选择select只需要在如下命令中同时按下`shift`故不赘述
   * winOS select all: `ctrl+a`
   * winOS next token: `ctrl+arrow`
   * winOS begin/end of line: `fn+arrow`
   * macOS select all: `command+a`
   * macOS next token: `option+arrow`
   * macOS begin/end of line: `command+arrow`, `fn+arrow` (only in vscode/chrome, not in wechat)
8. copy/cut/paste
   * winOS: `ctrl+c`, `ctrl+x`, `ctrl+v`
   * winOS history clipboard: `super+v`
   * macOS: `command+c`, `command+x`, `command+v`
   * macOS history clipboard: TODO software
9. (exit) full screen
   * winOS: `alt+space,x`, `alt+space,r`
   * macOS: double click, also double click to restore
   * macOS: `command+ctrl+f`, `command+ctrl+shift+f`
10. minimize窗口
    * winOS: `alt+space,i`
    * macOS: `command+m` 被缩放至Dock最右侧，不再被`command+tab`切换
11. logout, lock, sleep, shutdown
    * winOS: `win+x,u,i`, `win+l`, `win+x,u,s`, `win+x,u,u`
    * macOS: `command+shift+q`, `command+ctrl+q`, TODO, TODO
12. screenshot rectangular region to clipboard
    * winOS: `super+s`
    * macOS (modify setting): `shift+command+4`
13. 移动窗口
    * winOS: single tap, move
    * winOS: `alt+space,m`
    * macOS: single tap, move, single tap
    * macOS: TODO
14. 任务管理器，资源使用情况
    * winOS: `super+x,t`
    * macOS: install eul/stats, click the icon in action center
15. open file explorer (finder)
    * winOS: `win+e`
    * macOS: `option+command+space`
16. file explorer (finder) shortcut
    * winOS delete: `delete`
    * winOS new file/folder: TODO
    * macOS delete: `command+delete`
    * macOS new file/folder: TODO
17. remote
    * winOS enable ssh-server: setting, xxx, optional feature, openssh server
    * winOS GUI remote (xrdp): setting, xxx
      * (winOS) remote desktop
    * macOS enable ssh-server: setting, general, sharing, remote login
    * macOS GUI remote (VNC): setting, general, sharing, screen sharing
      * VNC (ipad user): install "Mocha VNC lite" (free for 5 minutes) or "Mocha VNC" (48 HKD)
      * VNC (mac user): screen sharing

macOS recommanded software

1. [github/awesome-mac](https://github.com/jaywcjlove/awesome-mac)
2. eul [github-link](https://github.com/gao-sun/eul)
3. stats [github-link](https://github.com/exelban/stats)
4. brew [link](https://brew.sh/)
   * `brew install git wget zsh tldr htop`
   * (win/linux user) `zsh` bindkey [link](https://apple.stackexchange.com/a/114528)
5. alfred
6. xcode
7. conda (see code snippet below) [link](https://docs.conda.io/en/latest/miniconda.html)
   * metal-related package: `tensorflow`, `pytorch` (see code block below), `taichi`

```bash
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

待发掘的功能

1. Stage manager: 功能与三指上划重叠
2. Widget: 两指从右侧边划入

possible bug

1. 三指上划的视图下，快捷键“command+space”依旧可以弹出搜索框，但无法输入任何字符
2. 快捷键“command+space”现弹出搜索框，然后三指上划，此时的视图下执行”command+space”快捷键没有任何反馈
