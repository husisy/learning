# git learning

1. link
   * [learngit-js](https://learngitbranching.js.org/)
   * [documentation](https://git-scm.com/book/en/v2)
   * [documentation-CN](https://git-scm.com/book/zh/v2)
   * [廖雪峰-git教程](https://www.liaoxuefeng.com/wiki/896043488029600)
   * `git help tutorial`
   * [知乎-为什么要先git add才能git commit](https://www.zhihu.com/question/19946553/answer/29033220)
   * [git cheat sheet](https://gitee.com/liaoxuefeng/learn-java/raw/master/teach/git-cheatsheet.pdf)
   * [github/progit/progit2](https://github.com/progit/progit2)
2. install
   * linux: `sudo apt install git`
   * windows: download from the offical site
   * `git config --global user.name husisy`: set global user name
   * `git config --global user.email "xxx@outlook.com"`: set global user email
3. concept
   * 工作区working directory，版本库Repository，暂存区，分支branch
   * untracked, unmodified, modified but not staged, modified and staged
   * commit-id
4. 常用命令
   * `git config --get user.name`: get global user name
   * `git config --get remote.origin.url`
   * `git help -a`: list all available subcommands. e.g. `git help pull`
   * `git help -g`: list all concept guides. e.g. `git help tutorial`
   * `git init`
   * `git add .`: current folder and subdirectories recursively
   * `git diff README.md`: compare unstaged with .git, 关于`diff`格式见[阮一峰-读懂diff](http://www.ruanyifeng.com/blog/2012/08/how_to_read_diff.html)
   * `git diff -cached README.md` compare staged with .git
   * `git status`
   * `git log -p`, `--patch`
   * `git log --stat --summary`
   * `git log --pretty=oneline`
   * `git log --graph --pretty=oneline --abbrev-commit`
   * `git reset --hard HEAD^`, `git reset --hard xxx-id`, `git reflog show`
   * `git reset HEAD xxx`，将暂存区中的`xxx`文件放回到工作区，删除暂存区的`xxx`文件
   * `git checkout -- xxx`，将工作区中的`xxx`文件回退倒版本库状态
   * `git rm xxx`，从版本库中删除`xxx`文件，同时从工作区中删除该文件
   * `git branch`，列出所有branch
   * `git branch xxx-branch`，创建`xxx-branch`
   * `git branch -d xxx-branch`，删除`xxx-branch`
   * `git checkout xxx-branch`，切换至`xxx-branch`，*TODO*替换为`git switch`，该命零在`git-2.23`加入
   * `git merge xxx-branch`，将`xxx-branch`合并至当前branch；相同文件不同行的编辑不产生merge冲突，用`git status`查看冲突内容
   * `git merge --no-ff -m "ignore" xxx-branch`，如之后不删除`xxx-branch`，则不建议使用fast forward（会将`xxx-branch`的操作记录合并到当前分支）
   * `git merge --abort`，当merge冲突时，用该命令返回merge前状态
   * `git stash`
   * `git stash list`
   * `git stash apply`, `git stash drop`
   * `git stash pop`
   * `git cherry-pick`
   * `git push remote-branch master-branch`
   * `git checkout dev origin/dev`
   * `git tag v1.0 9dfa`
   * `git branch --set-upstream-to=origin/dev dev`
   * `git rebase`
   * `git fetch`
5. git设计初衷
   * master分支应该是非常稳定的，仅用来发布新版本，平时只在`dev-branch`上干活
   * 多人合作则在`dev-branch`拉自己的分支，再不停地向`dev-branch`合并
6. `.gitignore`
   * [github/gitignore template](https://github.com/github/gitignore)
   * `git check-ignore -v xxx`
7. `.gitconfig`
8. 学习建议
   * 走通minimum working example-00，然后按“创建-添加-提交-修改-提交-修改-提交”顺序进行项目
   * 走通minimum working example-xx，使用github/gitee来掌握pull/push使用
   * 走通其它minimum working example

## minimum working example00

最终文件结构

```bash
ws00
├── README.md
└── .git
```

1. 在空文件夹`ws00`中初始化仓库 `git init`
2. 在`ws00`创建文件 `touch README.md`
3. 将文件添加至git仓库 `git add README.md`
4. 提交修改 `git commit --message "first commit"`

## minimum working example01 远程仓库

当测试远端仓库特性时，使用github/gitee可能不甚方便，故可考虑用本地克隆仓库来测试

git支持`git://`与`https://`协议来传输数据

1. 如使用`git://`协议，则需要在github/gitee网页上传个人的ssh-public-key，相关信息见`linux/bash` *TODO*
2. 如使用`https://`协议，则每次`git pull/push`时都需要输入用户名与密码
3. `git://`协议不走http/https代理，特殊网络环境需注意

场景一：本地现有仓库，将本地仓库同步至远端仓库

1. 网页创建空仓库
2. 本地`git remote add origin xxx.git`
   * 一般使用`origin`作为远端命名；如果使用多个远端，亦可命名为`github gitee`之类的
3. 本地执行`git push -u origin master`
   * `-u --set-upstream`

场景二：本地无内容，远端搭建仓库

1. 网页操作创建仓库
2. 本地`git clone xxx`

远端单分支使用场景 `origin/master`

1. `git clone xxx`
2. 本地修改文件然后`git commit -m "xxx"`
3. `git pull`
4. `git push origin master` 将本地修改

远端多分支使用场景 `origin/master origin/dev`

1. `git clone xxx`
2. `git checkout -b dev origin/dev` 此时本地`dev`关联`origin/dev`
3. `git push`将本地`dev`推送至`origin/dev`
4. `git checkout master`, `git merge dev`, `git push` 将本地`dev`合并至`master`，再将本地`master`推送至`origin/master`
   * `git checkout master`, `git merge origin/dev`, `git push`
5. `git branch -a -vv` 查看仓库关联情况

## minimum working example02 分支特性

1. `git init`
   * `touch README.md`
   * `git add README.md`
   * `git commit -m "first commit"`
2. `git branch dev`
   * `git checkout dev`
   * `"whatever" >> README.md`
   * `git commit -am "dev-branch commit"`
3. `git checkout master`
   * `git merge dev`
4. `git branch -d dev`

## misc

[stackoverflow/git-says-remote-ref-does-not-exist-when-i-delete-remote-branch](https://stackoverflow.com/a/35941658)

```bash
git fetch --prune
git push --delete origin test
```

```bash
# local remote0 remote1
# remote0/b0 -> remote1/b0
git fetch remote #local
git checkout b0
git push remote1

# https://stackoverflow.com/a/16756248
git config --global http.proxy 'socks5://127.0.0.1:23333'

# https://gist.github.com/rbialek/1012262
git remote set-url
```

## zcnote

why git, 当代码量超过1000行时，git能够有效保证你的代码依旧能有效被组织

1. 如果你的工作场景中有大量的时间在git相关命令上，那多半是git的使用方式不对
2. 为什么有stage这个状态：使用起来像是图形界面的打勾
3. 不要给初学者讲git的原理，从git的使用场景开始讲起
