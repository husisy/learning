# minimum working example 使用说明

1. 安装：`pip install .`
2. 运行：
   * 在命令行执行`python -c "from zctest_link import pyadd; print(pyadd(2,33))"`，**不要**在当前路径执行，当前路径的package具有更高优先级
3. 卸载：`pip uninstall zctest_link`

```bash
g++ -c src/utils.cpp -std=c++11 -I include -o lib/utils.o
```
