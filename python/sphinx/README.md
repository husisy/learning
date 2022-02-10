# sphinx

1. link
   * [official site](http://www.sphinx-doc.org/en/master/)
   * [quickstart](http://www.sphinx-doc.org/en/master/usage/quickstart.html)
   * [sphin language support](https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language)
   * [a brief tutorial on sphinx and reStructured text](https://iridescent.ink/HowToMakeDocs/index.html)
   * [中文搜索问题](https://iridescent.ink/HowToMakeDocs/Basic/Sphinx.html#secchinesesearchproblem)
   * [vscode-extension-restructuredtext](https://docs.restructuredtext.net/index.html)
   * [Docutils-documentation](https://docutils-zh-cn.readthedocs.io/zh_CN/latest/index.html)
2. install
   * `conda install -c conda-forge sphinx sphinx-autobuild`
   * `pip install sphinx sphinx-autobuild`
   * `pip install rstcheck`
3. `sphinx-quickstart`

```bash
sphinx-quickstart
sphinx-build -b html source build #make html
# sphinx-build -b latexpdf source build
```

TODO

1. learn reST first, see [link](https://iridescent.ink/HowToMakeDocs/Basic/reST.html#restructuredtextsimpletutorial)
