# pytest

## overview

1. link
   * [documentation](https://docs.pytest.org/en/latest/)
   * [github](https://github.com/pytest-dev/pytest/)
2. install
   * `conda instll -n conda-forge pytest`
   * `pip install pytest`
3. 常见用法
   * `pytest --version`: 测试是否安装成功
   * `pytest`
   * `pytest xxx.py`: test single file
   * `pytest xxx.py::test_hf0`: test specific function
   * `pytest --maxfail=2`: stop after two failures
4. [standard test discovery rules](https://docs.pytest.org/en/latest/goodpractices.html#test-discovery)
   * current directory and subdirectories
   * param: `testpaths`
   * param: `norecursedirs`
   * file: `test_*.py`, `*_test.py`
   * item: `test` prefixed function outside of class, `test` prefixed function inside `Test` prefixed test classes (without an `__init__` method)
5. project file structure
   * [packaging a python library](https://blog.ionelmc.ro/2014/05/25/python-packaging/)
6. fixture
   * `tmpdir`: unique temporary directory
7. mark: `pytest -m slow`, `@pytest.mark.slow`
8. run test from package: `pytest --pyargs pkg.testing`
