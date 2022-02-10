# course

link: [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)

此处仅包含chap1与chap2，其余章节见`klearning`仓库

## chap1 课程概览与shell

1. bash界面 `doge@laptop:~$` 表示用户名`doge`，设备名hostname `laptop`，当前工作目录`~`（即home目录），`$`表示bash正在等待输入且当前用户并非root
   * 一般用户名**不可以**包含大写字母
   * current working directory (cwd)
   * root用户提示符`#`
2. 常用命令`date, echo`
3. 获得可执行文件路径`which echo`
4. 环境变量
   * `PATH`
   * `SHELL`
5. 文件系统
   * 绝对路径，相对路径
   * 特殊目录：当前目录`.`，上级目录`..`，home目录`~`，历史工作目录`-` (OLDPWD)
   * 打印当前路径`pwd`
   * 打印当前目录文件`ls`
   * 打印文件内容`cat hello.txt`
   * 文件权限：读`r`，写`w`，执行`x`
   * 文件夹权限：`r`打印文件夹中文件，`w`在文件夹中编辑文件，`x`切换至该目录
   * 打印文件属性`ls -l`：所有者/用户组/其他人分别具有的权限
   * 切换当前路径`cd ..`
   * 创建文件`touch hello.txt`
   * 修改文件权限`chmod ugoa+-rwx`
   * 创建文件夹`mkdir hello`
   * 重命名`mv hello.txt world.txt`
   * 删除文件`rm world.txt`
6. 获得帮助
   * `ls --help`
   * `man ls`：进入页面输入`q`退出
   * `tldr`
   * `man -k xxx`：已知部分关键字
   * `whatis xxx`：简要说明
   * `info xxx`：详细说明
   * `apropos`
   * `help true`：查询bash built-in commands
   * `which xxx`：完整路径
   * `whereis xxx`：搜索路径
7. 输入输出流，重定向`><`，追加内容`>>`，管道`|`
8. root用户，管理员权限
   * UID=0
   * `sudo`: Switch User (SU) to do it，默认root
   * 切换至root用户：`su root`, `sudo -i`
   * 重定向命令的执行者是当前用户，即sudo不跨越重定向命令
9. 快捷键shortcut
   * `ctrl-U`：将当前输入放入粘贴板
   * `ctrl-W`：将光标前一个单词放入粘贴板
   * `ctrl-Y`：粘贴
   * `ctrl-R`：反向搜索
   * `ctrl-C`：Interrupt
   * `ctrl-L`：清屏
   * `up_arrow`：遍历命令
   * `ctrl-D`: end of line (EOL)
   * `tab`：自动补齐
10. 系统挂载在`/sys`目录
11. 字符串转义 [GNU-bash-quoting](https://www.gnu.org/software/bash/manual/html_node/Quoting.html#Quoting)
    * `'$foo'`不转义，`"$foo"`转义

```bash
date
echo hello
echo "My Photos"
echo My\ Photos

which echo
echo $PATH
echo $SHELL

# filesystem
ls
pwd
cd /home
cd ..
mv
cp
mkdir

# get help
ls --help
man ls #type q to quit

echo "hello world" > tbd00.txt
cat tbd00.txt
cat < tbd00.txt
cat < tbd00.txt > tbd01.txt
echo "world hello" >> tbd00.txt
ls -l | tail -n1

cat /sys/class/power_supply/BAT1/capacity #get battery power percent
cat /sys/class/thermal/thermal_zone0/temp #get cpu temperature

curl --head --silent google.com | grep -i content-length | cut --delimiter=" " -f2
```

## chap2 Shell工具和脚本

1. 变量赋值`foo=bar`：等号两边不能有空格
2. 函数参数
   * `$0`：脚本名、函数名
   * `$1 $2 ... $9`：第x个参数
   * `$#`：参数个数
   * `$@`：所有参数
   * `$?`：前一个命令的返回值，`0`表示正常退出
   * `$$`：当前进程识别码PID
   * `!!`：完整的上一条命令，包括参数。常见应用：当你因为权限不足执行命令失败时，可以使用`sudo !!`再尝试一次
   * `$_`：上一条命令的最后一个参数
3. 双目逻辑运算符`&& ||`：短路运算符
   * `true`返回值是`0`
   * `false`返回值是`1`
4. 命令替换command substitution `echo $(date)`
5. 进程替换 `diff <(ls foo) <(ls bar)`
6. globbing 通配符`* ? {}`
7. `shellcheck`工具 [github/shellcheck](https://github.com/koalaman/shellcheck)
8. 查找: `find fd locate grep ack ag rg`
9. 文件夹导航`fasd autojump`
   * `ls -R`
   * `tree`
   * `broot`
10. 运行bash scripts
    * `source xxx.sh`, `. xxx.sh`
    * `bash xxx.sh`
11. 命令查找
    * `up/down` arrow
    * `history`
    * `ctrl+R`
    * `fzf`
    * `zsh`

```bash
foo=bar
# foo = bar fail

# globbing
touch {foo,bar}/{a..h}
mv /path/to/project/{foo,bar,baz}.sh /newpath

ls > /dev/null 2> /dev/null

# shebang
/usr/bin/env python

# find
find . -name src -type d
find . -path "*/test/*.py" -type f
find . -mtime -1
find . -size +500k -size -10M -name "*.tar.gz"
fd

locate draft00.py
udpatedb

grep
rg -u --files-without-match '^#\!' -t sh

ls -lt #sorted
ls --color
```
