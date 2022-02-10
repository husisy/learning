# linux make

1. link
   * [阮一峰/Make命令教程](http://www.ruanyifeng.com/blog/2015/02/make.html)
   * [github/gist/isaacs/Makefile](https://gist.github.com/isaacs/62a2d1825d04437c6f08)
   * [gnu make manual](https://www.gnu.org/software/make/manual/make.html)
2. **必须**使用tab, `.RECIPEPREFIX`
3. 默认`makefile`，默认执行第一个目标
4. echo每一行执行的命令，包括`#`开头的注释代码，以`@`开头的行不echo
5. 每行命令commands在单独的进程中进行，因此不在一行的export没有作用，在同一行的话，export的必要性不大
6. 特殊变量
   * `$@`, `$(@D)`, `$(@F)`: target, target dir, target file
   * `$<`, `$(<D)`, `$(<F)`: the first prerequisite in the list
   * `$^`: the prerequisite list
   * `$?`：所有相比目标文件时间更新了的文件
   * `$$`：bash `$`
   * `$*`
   * `.PHONY`: 非文件变量，永远都会执行

```bash
.RECIPEPREFIX = $
.PHONY: clean all

tmp_a.txt:
$ touch tmp_a.txt
tmp_b.txt:
$ echo "tmp_b.txt" > tmp_b.txt
tmp_c.txt: tmp_b.txt
$ @# close echoing
$ cp tmp_b.txt tmp_c.txt

all: tmp_a.txt tmp_b.txt tmp_c.txt
clean:
$ rm tmp_*.txt
```
