# learning

开心就好

## 仓库内容与组织方式

该仓库包含各种计算机技能的代码片段以及个人笔记，并且按照编程语言进行归类组织，包括但不限于

```bash
.
├── python
│   ├── python
│   ├── pytorch
│   ├── tensorflow
│   └── scipy
├── linux
│   ├── docker
│   ├── git
│   └── shell
└── cpp
    ├── cpp
    ├── STL
    ├── cmake
    └── boost
```

通常一级目录夹表示编程语言，二级目录表示基于该语言的库。这些文件夹下的文件通常可以按照文件名推断其用处

1. `xxx.md`：说明性文字，例如`README.md`通常会包含该库的官网、文档等链接，安装命令以及最小可运行代码示例minimum working example (MWE)
2. `draft_x.y`: 代码片段，例如`draft00.py`通常是一系列的代码片段，建议在交互式环境下（例如`ipython/bash`）运行并观察结果
3. `demo_x.y`：函数封装的代码片段，通常由`draft_x.y`整理而来，例如`python/matplotlib/demo_3d.py`展示三维图像绘制
   * 但并非所有的`draft`都适合整理为`demo`
4. `test_x.y`：单元测试代码片段，通常由`demo_x.y`整理而来，例如`python/scipy/test_linalg.py`中会测试`svd`分解得到的`U`矩阵是幺正的
   * 调试特定库的错误时，我有时会运行一遍这些`test_x.y`文件（命令行下输入`pytest -v xxx`即可）来确保库的安装是正确的
5. `ws_xx/`文件夹：当一个功能无法用一个`draft_x.y`文件来展示时，便放在单独的文件夹下
   * `ws`是`workspce`的缩写

## 如何使用该仓库

1. 学习某个具体的库：找到该库所在的目录，先阅读`README.md`，其中包含参考链接以及安装命令，然后依次是`draft_x.y`运行一系列的代码片段。这些代码片段往往是自明的，但也因为缺少注释而显得晦涩难懂，建议同时阅读`README.md`中记录的官方文档
2. 从最小代码片段构建项目代码：`draft_x.y`中包含一系列的最小可运行代码片段，作为构建项目的基础砖块再合适不过
3. 对项目代码中不确定的代码行为，编写最小运行代码来确定其行为，并记录于该仓库
4. 构建自己的笔记系统：每个人的知识体系差异巨大，所需要的笔记系统自然也会很不一样。笔者认为重要的工具也许完全不适用于其他领域，但笔记的组织方式应该是通用的。因此笔者强烈建议读者开始建立自己的笔记系统

## 其他

1. 该仓库**不**接受pull request
   * 该仓库的组织方式以及内容很大程度上是个人风格决定的，且很难达成共识（行宽，空格，换行等）
   * 每个人的笔记系统会很不一样，我很难想象有人会在该仓库的基础上进一步添加内容，也许重头建立个人专属的笔记系统会更合适
   * 如果你一定需要在该仓库的基础上进一步添加内容，直接fork该仓库并添加内容就好了
2. 关于license
   * 选择license是一件极其头疼的事情，见[choose-a-license](https://choosealicense.com/) [开源指南](https://opensourceway.community/open-source-guide/legal/)等。直到我看到这个Linus对于GNUv2协议的解释「我給你源代碼，你給我你對它的修改，我們就扯平了I give you source code, you give me your changes back, we are even.」，我很喜欢这个简洁的解释，所以我选择GNUv2
   * 必须承认的是，这个仓库的大部分文字是“偷”来的。通过阅读其他仓库的源码和文档，吸收其中的关键部分组成这个仓库，最糟糕的是部分代码只是复制粘贴而无理解吸收（明显的抄袭行为），但愿之后我会有空整理这类代码。对于其中license不兼容的文字，恳请告知，我会将其删除

## TODO

1. [ ] 向前向后兼容性问题
2. [ ] python-qt5
3. [ ] indexing and kernel
4. [ ] `python/deepxde`
5. [ ] [github/RLcode](https://github.com/louisnino/RLcode)
6. [ ] [github/tvm-learn](https://github.com/BBuf/tvm_learn)
7. [ ] [github/sisl](https://github.com/zerothi/sisl): python tight-binding, NEGF, DFT
8. [ ] quantum chemistry package
9. [ ] pde: [github/solver-in-the-loop](https://github.com/tum-pbs/Solver-in-the-Loop)
