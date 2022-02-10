# spyder

1. link
   * [ipython doc](http://ipython.readthedocs.io/en/stable/index.html)
   * [ipythonconsole](https://pythonhosted.org/spyder/ipythonconsole.html)
   * [peps-0008](https://www.python.org/dev/peps/pep-0008/)
2. 一个问号只是显示对象的签名，文档字符串以及代码文件的位置，二个问号可以直接显示源代码
3. 调用系统Shell命令: `!ls`, `!pwd`
4. tab自动补全
5. 历史记录。IPython把输入的历史记录存放在个人配置目录下的history.sqlite文件中，并且可以结合%rerun、%recall、%macro、%save等Magic函数使用。尤为有意义的是，它把最近的三次执行记录绑定在_、__和___这三个变量上。搜索历史记录时，还支持Ctrl-r、 Ctrl-n和Ctrl-p等快捷键
6. `%debug`：激活交互的调试器
7. `%hist` 或者 %history：查看历史纪录
8. `%load`：加载外部代码
9. spyder code cell: `#%%`
