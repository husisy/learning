# shell

1. link
   * [the linux command line for beginners](https://tutorials.ubuntu.com/tutorial/command-line-for-beginners#0)
   * [github-linuxtools](https://github.com/me115/linuxtools_rst)
   * [udacity - Linux Command Line Basics](https://classroom.udacity.com/courses/ud595)
   * [gitbook-shell编程范例](https://tinylab.gitbooks.io/shellbook/zh/preface/01-chapter1.html)
   * [github-Shell脚本编程30分钟入门](https://github.com/qinjx/30min_guides/blob/master/shell.md)
   * [阮一峰-xargs命令教程](https://www.ruanyifeng.com/blog/2019/08/xargs-tutorial.html)
   * [Shell脚本调试技术](https://www.ibm.com/developerworks/cn/linux/l-cn-shell-debug/index.html)
   * [MIT/The Missing Semester of Your CS Education](https://missing.csail.mit.edu/)
   * [advanced-bash-scripting-guide](https://tldp.org/LDP/abs/html/)
   * [github/tldr](https://github.com/tldr-pages/tldr)
   * [CLI-github](https://cli.ninghao.net/)
   * [explainshell](https://explainshell.com/)
2. concept
   * command line interface (CLI)
   * terminal (terminal emulator): a program that draws text in a window and let you type things in on a keyboard
   * shell: `sh`, `GNU bash`, `TCSH`, `KSH`, `Seashell`
   * console, prompt
3. `man`页面所属分类标识
   * (1) 用户可以操作的命令或者是可执行文件
   * (2) 系统核心可调用的函数与工具等
   * (3) 一些常用的函数与数据库
   * (4) 设备文件的说明
   * (5) 设置文件或者某些文件的格式
   * (6) 游戏
   * (7) 惯例与协议等。例如Linux标准文件系统、网络协议、ASCⅡ，码等说明内容
   * (8) 系统管理员可用的管理条令
   * (9) 与内核有关的文件
4. 个人偏见
   * `less`替代`more`
   * 使用`. ./xxx.sh`而不使用`source ./xxx.sh`
5. redirecting
   * `ll > test0.txt`
   * `echo "this is a test" > test1.txt`
   * append `cat test* >> combined.txt`
6. plumbing管道, named pipe
   * 符号`|`前后加上空格（虽然不加空格行为不变）
   * `ls ~ > draft.txt` -> `wc -l draft.txt` -> `rm draft.txt`; `ls ~ | wc -l`
   * `sort draft.txt | uniq | wc -l`
7. bash script：命令行参数
8. 显示当前shell `echo $SHELL`, `echo $$`, `ps -C bash`
9. 上一次进程的结束状态（退出状态码） `echo $?`
10. `source xxx.sh`, `. xxx.sh`
11. 冒号开头的语法 [stackoverflow](https://stackoverflow.com/a/32343069)
12. parameter expansion [bash-backers-wiki](https://wiki.bash-hackers.org/syntax/pe) [stackexchange](https://unix.stackexchange.com/a/122848)
13. shortcut: `ctrl+k/u/w`, `ctrl+y`, `ctrl+r`, `ctrl+l`
14. 配置文件 `.bash_profile`, `.profile`
15. `su`: switch user
16. grub mode
17. array variables [tldp-bash-guide-for-beginners](https://tldp.org/LDP/Bash-Beginners-Guide/html/sect_10_02.html)

```bash
type ls
type which
type cd
help cd
echo $RANDOM
info ls
```

```bash
who
netstat -a
ps -aux
sync
shutdown reboot halt poweroff; shutdown -h now
fsck /dev/sda7
whoami
hostname

reset
uniq
sort

sudo useradd x
sudo passwd x
sudo gpasswd -a x wheel

curl --help
man curl #shortcut / n N f b q

mkdir -p awesome_project/app/styles
touch README.md
echo 'CLI'  >> README.md
cat README.md

rsync

uname -m && cat /etc/*release
chmod u+x g-w o-w file #ugoa User Group Other All; rwx

chown
usermod -aG
groupsadd
groups
# /etc/group
```

## 非基础命令：高级命令，非核心命令

1. `date`
2. `bc`: basic calculator
3. `uname`
4. `hostname`
5. `uptime`
6. `history`
7. `host`
8. `echo`
9. `expr`
10. `bash`

## file system

```bash
cd #home
cd ~ #home
cd - #change directory to last work directory
find . | wc -l #count item in currect directory (include . and ..)
alias lsl='ls -trlh'
alias lm='ls -al|more'
find ./ -name "core*" | xargs file
find ./ -name "*.o" -exec rm {} \;
```

## 用户权限系统

1. superuser, `su`, `sudo`
   * ubuntu has disabled the `root` account `sudo su`, the first user created is considered to be the superuser
   * (ubuntu) `/etc/shadow`: encrypted passwords
2. `adduser` (推荐) 与 `useradd`：[stackexchange](https://askubuntu.com/questions/345974/what-is-the-difference-between-adduser-and-useradd)
3. `groupadd userdel groupdel chown`
