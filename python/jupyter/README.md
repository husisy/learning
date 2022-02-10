# Jupyter

1. link
   * [documentation](https://jupyterlab.readthedocs.io/en/stable/)
   * [github](https://github.com/jupyterlab/jupyterlab)
2. JupyterLab interface
   * main work area
   * left sidebar
   * menu bar
   * file browser
   * list of running kernels and terminals
   * command palette
   * notebook cell tools inspector
   * tabs list
3. jupyter server
   * 生成配置文件`jupyter notebook --generate-config`
   * 设置密码`jupyter notebook password`
4. matplotlib相关配置
   * 推荐使用`%matplotlib notebook`而非`%matplotlib inline`
5. server jupyter notebook：见`linux/linux/README_software.md/ssh`端口转发部分
   * [跑深度学习代码在linux服务器上的常用操作(ssh,screen,tensorboard,jupyter notebook)](https://zhuanlan.zhihu.com/p/31457591?utm_source=wechat_session&utm_medium=social)
   * [如何在window访问ubuntu服务器的jupyter notebook](https://zhuanlan.zhihu.com/p/30845372)
   * [设置 jupyter notebook 可远程访问](http://www.jianshu.com/p/444c3ae23035)
   * [official documentation](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html#notebook-server-security)

```bash
conda install -c conda-forge jupyterlab
jupyter notebook --version
jupyter lab --no-browser --port 8888 --ip 127.0.0.1
```

## jupyter-book

1. link
   * [github](https://github.com/executablebooks/jupyter-book)
   * [documentation](https://jupyterbook.org/index.html)
2. install
   * `pip install jupyter-book`
   * `cnoda install -c conda-forge jupyter-book`
3. jupytext
4. MyST Markdown：方便了展示，方便了git管理，但不方便运行

```bash
jupyter-book create mynewbook
cd mynewbook
jupyter-book build . #--all
pip install ghp-import
ghp-import -n -p -f _build/html
```
