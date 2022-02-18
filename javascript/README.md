# javascript

1. link
   * [廖雪峰](https://www.liaoxuefeng.com/wiki/001434446689867b27157e896e74d51a89c25cc8b43bdb3000)
   * [mozilla-js-tutorial](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript)
2. 相关
   * 网景公司Netscape Communications - Navigator浏览器 - JavaScript
   * Microsoft - JScript
   * European Computer Manufacturers Association (ECMAScript)
3. chrome console: right click, `inspect`, `console`
4. 偏见
   * 所有语句**必须**添加`;`
   * **禁止**使用`==`，建议使用`===`，不建议使用`==`；`===`不会自动类型转换，数据类型不一致返回`false`
   * `undefined`**仅**用于在判断函数参数是否传递，其余情况都使用`null`
   * **必须**使用`'use strict';`
   * **禁止**对字符串下标赋值，虽然不报错
   * **禁止**对数组越界下标赋值，**禁止**修改数组`length`属性
5. 变量名：大小写字母、数字，`$_`，非数字开头，非关键字
6. 字符串是不可变的

## nvm

1. link
   * [github/nvm](https://github.com/creationix/nvm)
   * [github/nvm-windows](https://github.com/coreybutler/nvm-windows)

```bash
nvm install 16 #install before use it
nvm list
nvm use 16.14 #administrator required on windows

npm install -g yarm
```
