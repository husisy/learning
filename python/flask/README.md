# flask

1. link
   * [github](https://github.com/pallets/flask)
   * [documentation](https://flask.palletsprojects.com/en/1.1.x/)
2. 安装
   * `conda install -c conda-forge flask coverage`
   * `pip install flask coverage`
3. 特殊环境变量
   * `export FLASK_APP=draft00.py`, win-cmd `set FLASK_APP=draft00.py`, powershell `$env:FLASK_APP="draft00.py"`
   * 调试模式`export FLASK_ENV=development`
4. 偏见
   * 使用`@app.route('/about')`，**禁止**使用`@app.route('/about/')`
5. 启动命令
   * `flask run`
   * `--host=0.0.0`
   * `--port=2333`
6. 变量规则
   * `string`: （缺省值） 接受任何不包含斜杠的文本
   * `int`: 接受正整数
   * `float`: 接受正浮点数
   * `path`: 类似`string`，但可以包含斜杠
   * `uuid`: 接受`UUID`字符串
7. HTTP方法：`GET,POST,HEAD,OPTIONS`
