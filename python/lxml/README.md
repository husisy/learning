# lxml

1. link
   * [official site](https://lxml.de/): tutorial, FAQ
   * [lxml tutorial](https://lxml.de/tutorial.html)
   * [XPath - w3school](http://www.w3school.com.cn/xpath/index.asp)
   * [XPath tutorial - ZVON.org](http://zvon.org/comp/r/tut-XPath_1.html#intro)：**网站已关闭**
   * [Concise XPath](http://plasmasturm.org/log/xpath101/)
2. installation
   * `conda install -c conda-forge lxml`
   * `pip install lxml`
   * 如使用css：`conda install -c conda-forge cssselect`, `pip install cssselect`
3. usage: `from lxml import etree`
4. XPath使用路径表达式：选取xml文档中的节点或节点集
5. Extensible Stylesheet Language Transformations (XSLT)
6. XQuery, XPointer
7. 节点类型
   * 元素
   * 属性
   * 文本
   * 命名空间
   * 处理指令
   * 注释
   * 文档节点（根节点）
8. 项目item
   * 基本值atomic value：无父或无子的节点
   * 节点

## misc

1. `/AA/BB/CC`: absolute path
2. `AA/BB/CC`, `./AA/BB/CC`: relative path
3. `//AA/BB/CC`: start anywhere
4. `//AA/@bb`: attribute
5. `[]`: condition
   * `[1]`, `[last()]`, `[last()-1]`: indexing
   * `[@aa]`, `[@aa="233"]`: attribute
   * `[@*]`, `[not(@*)]`
6. misc function
   * `[name()="233"]`
   * `[starts-with(name(), "233"]`
   * `[contains(name(),"233")]`
   * `[count(*)=2]`
   * `[string-length(name())=3]`
7. `/AA | /BB`: or
8. `::`: axis, `child::` is the default axis
   * `child::`
   * `descendant::`
   * `parent::`
   * `ancestor::`
9. numeric operations
   * `[position() mod 2 = 0]`
   * `[position() = floor(last() div 2 + 0.5) or position() = ceiling(last() div 2 + 0.5)]`
