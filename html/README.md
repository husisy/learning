# html tutorial

该文件夹自身即用于html学习，可以预期不会创建以html为主体的项目。

## overview

1. Hyper Text Markup Language (HTML) 超文本标记语言
2. html元素：HTML标签通常是成对出现的
   * 开放标签opening tag
   * 元素内容closing tag
   * 闭合标签
3. HTML常见元素
   * `<html></html>`
   * `<body></body>`
   * `<h1></h1>`, `<h6></h6>`
   * `<p><p>`
4. html空元素
   * 使用`<br />`，禁止使用`<br>`；在xhtml, xml以及未来版本的html中，所有元素必须关闭：在开始标签中添加斜杠
   * 使用`<p></p>`，禁止使用`<P></P>`；html标签、属性、属性值对大小写不敏感，但未来的版本中可能强制使用小写
   * 始终为属性值加单引号/双引号
5. 属性
   * `<h1 align="center"></h1>`
   * `<body bgcolor="#c4c4c4"></body>`
   * `<table border="1"></table>`
   * [html属性参考手册](http://www.w3school.com.cn/tags/index.asp)
6. html自动在块级元素前后添加一个额外的空行，比如段落、标题元素
   * html heading子用于标题，不要仅仅是为了产生粗体或大号的文本而使用heading
   * 搜索引擎使用标题为网页的结构和内容编制索引
7. 所有的换行都视为空格，所有的连续空格都算作一个空格，即`%\t \n%`等价于`% %`（不包含开头与结尾的百分号）。
8. 使用`style`替代如下标签或属性
   * `<center>`, `<font>`, `<basefont>`, `<s>`, `<strike>`, `<u>`
   * `align=""`, `bgcolor=""`, `color=""`
9. 子文件夹的hyperlink请始终添加`/`，例如`husisy.com/misc/`而不是`husisy.com/misc`（假设`misc`是一个文件夹）

## CSS

1. [W3Cschool - css教程](https://www.w3cschool.cn/css/)
2. Cascading Style Sheets层叠样式表
   * 标记语言，属于浏览器解释型语言，直接由浏览器执行，不需要编译
   * 用来表现HTML/XML
   * 由W3C的CSS工作组发布推荐和维护
   * 构成：选择器selector、属性property、值value
3. 优势
   * 减少网页的代码量
   * 增加网页的浏览速度
4. 外联式linking（外部样式）、嵌入式embedding（内页样式）、内联式inline（行内样式）
5. 颜色
   * 英文单词：`p {color: red;}`
   * 十六进制：`p {color: #ff0000;}`
   * RGB值：`p {color: rgb(255,0,0);}`, `p {color: rgb(100%,0%,0%)}`
6. 注释：`/*this is a comment*/`
7. id选择器
   * `<p id="para1">hello world</p>`
   * `#para1 {color:red;}`
   * 非数字开头，当前html文件中唯一标识
8. class选择器
   * `<p class="center">hello world</p>`
   * `.center {text-align:center;}`
   * `p.center {text-align:center;}`
   * 非数字开头
