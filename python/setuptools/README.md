# setuptools

1. link
   * [documentation](https://setuptools.readthedocs.io/en/latest/index.html)
   * [pypi-packaing-tutorial](https://packaging.python.org/tutorials/packaging-projects/)
   * [documentation/developer-guide](https://setuptools.readthedocs.io/en/latest/setuptools.
2. 偏见
   * `setup(name=xxx)`中**禁止**有下划线`_`、连接符`-`，python-module-name中可以使用
   * 使用`setup(entry_points=xxx)`而非使用`setup(scripts=xxx)`
   * 无须使用`pkg_resources`，python eggs is deprecated
   * **必须**做一层`python/xxx-module`文件隔离，避免当前路径而非`site-package`导入问题
3. 单文件（Module）使用`setup(py_moduels=[xxx])`，见[github-issue](https://github.com/pypa/packaging.python.org/issues/397)
4. 暂且只管`install_requires`，不管`setup_requires test_requires`
5. 名词
   * version
   * bootstrap module
   * Python Eggs **deprecated**
   * namespace package
   * `easy_install` **deprecated**
6. 对于`scripts`的使用，`#! python`必须放首行，否则会出现奇奇怪怪的`unable to open X server`错误

```Python
# license: https://choosealicense.com/
# python setup.py sdist bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# python setup.py clean --all
# pip install --index-url https://test.pypi.org/simple/ husisy_my_pkg
```

## ws00

1. 安装 `pip install -e .`
2. 运行 `python -c "import mypackage; mypackage.say()"`
3. 卸载 `pip uninstall zcpackage`

## pip

1. `pip install xxx`
   * `pip install xxx.whl` 文件下载后安装
   * `pip install .` 从源文件（含`setup.py`）安装
   * `pip install -upgrade xxx`
2. `pip show xxx`
   * `pip show --files xxx` Package安装的文件
3. `pip list`
   * `pip list --outdated`
4. `pip download xxx`
5. `pip wheel xxx`
6. `pip uninstall xxx`
7. 建议：由于linux系统接管了python以及pip，普通无权限操作pyhton/pip本身（也不建议），故建议使用miniconda创建一份完全属于用户自己的pyhton环境（含pip）
8. specify package version
   * `pip install xxx==1.0.1`
   * `pip install 'xxx>1.0.2`, 因为`>`是bash中的特殊符号，此处使用引号转义
   * `pip install xxx>=1.0.3`
9. 代理
   * `--proxy [user:passwd@]porxy.server:port`
   * 设置环境变量 `http_proxy https_proxy no_proxy`
10. `pip install -r requirements.txt`, see [Requirements File Format](https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format)
    * `pip freeze > requirements.txt`
    * pip does NOT have a true dependency resolution

## Packaging Python Projects

1. link
   * [Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects)
   * [packaging and distributing projects](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
2. `setup.py`
   * `name`: uid
   * `version`: [PEP440](https://www.python.org/dev/peps/pep-0440/)
   * `author`, `author_email`
   * `description`: a short, one-sentence summary
   * `long_description`, `long_description_content_type`
   * `url`
   * `packages`
   * `classifiers`: see [list of classifiers](https://pypi.org/classifiers/)
3. `README.md`
   * see [github flavored markdown](https://guides.github.com/features/mastering-markdown/)
4. `LICENSE`
   * see [choosealicense](https://choosealicense.com/)
5. `python setup.py sdist bdist_wheel`
   * `dist/xxx.tar.gz`
   * `dist/xxx.whl`
6. [test pypi](https://test.pypi.org/)
   * `python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*`

```bash
# folder structure
ws01/
|-- /example_pkg
|   |-- __init__.py
|-- setup.py
|-- LICENSE
|-- README.md
```
