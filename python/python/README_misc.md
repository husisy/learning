# miscellaneous

## 个人约定

1. 所有约定必须是可修改的。不开心的约定，改就是了
2. 长篇的代码应使用空行分隔为若干个代码块，必要时在代码块的首行添加注释
3. 单行代码长度可以很长。逻辑简单的单行代码，即使很长亦可接受
4. 逻辑复杂的单行代码应该分成若干片段表达式构成单个代码块，并存放于`tmp0,tmp1`这些临时变量名中，绝对禁止临时变量名跨代码块
   * `x,y,z, xxx_i, x0,x1,x2`: 临时变量名，常用于循环变量
   * `zc0,zc1,zc2`：仅用于个人代码，例如代码调试、代码重构等场景
   * `ind0,ind1,ind2`：MATLAB编程留下的命名习惯，由于MATLAB中的indexing极其重要（缺少必要的`zip(),enumerate()`函数），而这些indexing又是作为循环变量跨若干个代码块，关于indexing的任何操作必须异常谨慎
5. `hf0,hf1`：用于匿名函数命名，取名源自于MATLAB编程中的`Function Handle`
6. `generate_hf_`：用于生成Python中匿名函数，由于Python的closure作用域特性
7. 当且仅当有意义方才赋予有意义的变量名

## git管理python项目

link

1. [git文档](https://git-scm.com/docs/gitignore)

git代码仓库中往往会存在一些不适合git同步的信息，例如

1. 开发者个人文件：例如开发者可能会将与该项目相关的文件（例如参考文献）放入项目文件夹
2. 账号密码：服务器ssh登录账号、数据库登录等

对于前者采取的策略是，在`.gitignore`文件中添加`_developer/`，那么在`_developer/`文件夹下的所有文件便不参与git同步，更多的`.gitignore`用法见

对于后者采取的策略是，使用[规范配置文件格式-wiki](https://zh.wikipedia.org/wiki/配置文件)并将该配置文件加入`.gitignore`文件中。常见的配置文件格式有`.ini .json .yaml .toml`。个人采用的是`.ini`格式，相关的文件有

1. `project_dependency.ini`：详细`.ini`文件格式见[wiki](https://en.wikipedia.org/wiki/INI_file)
   * 所有的账号密码信息
   * 其他可能的全局变量（例如`development/deployment`）
   * 该文件不参与git同步
2. `project_dependency.ini.example`
   * 与`project_dependency.ini`参数一致但不包含具体的值，开发者可根据该文件补全相应的`project_dependency.ini`文件
   * 该文件参与git同步
3. `project_dependency.py` 一般Python脚本
   * 检查项目文件结构是否完整
   * 加载`project_dependency.ini`中的参数
   * 将参数注入到对应的模块中去（开发者可直接使用模块提供的接口而无需考虑具体的账号密码信息）
   * 命令行下查看所有参数 `python -c "from project_dependency import get_config; get_config()"`
