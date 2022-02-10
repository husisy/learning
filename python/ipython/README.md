# ipython

1. link
   * [official site](https://ipython.org/)
   * [ipython documentation](https://ipython.readthedocs.io/en/stable/)
   * [ipython github wiki](https://github.com/ipython/ipython/wiki?path=Cookbook)
   * [jupyter documentation](https://jupyter.readthedocs.io/en/latest/)
   * [python prompt toolkit documentation](https://python-prompt-toolkit.readthedocs.io/en/stable/)
   * [jupyter widgets documentation](https://ipywidgets.readthedocs.io/en/stable/)
2. 特性
   * tab-completion
   * object introspection: `abs?`, `abs??`
   * magic command
   * system shell access
   * command history retrieval
3. get help
   * `?` introduction and overview of IPython's feature
   * `object?` `object??` details about `object`
   * `%quickref` quick reference
   * `help` python's own help system
4. shortcuts
   * `Ctrl-o`换行
5. configuration
   * `ipython profile create`

## tab completion

```bash
>>> data = ['233', 233]
... data[0].<tab>
```

## magic command

1. line magics and cell magics
2. get help
   * `%run?`
   * `%magic`
   * `%lsmagic`
3. automagic默认启用，可省略`%`
4. 常用命令：`%whos %run %cd %timeit %debug %magic %quickref %alias`
5. 弃用`%matplotlib`，用`plt.ion() plt.ioff()`替代
