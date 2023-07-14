# markdown

1. link
   * [阳志平的网志](http://www.yangzhiping.com/)
   * [Learning-Markdown (Markdown 入门参考)](http://xianbai.me/learn-md/index.html)
   * [Markdown 语法说明](http://wowubuntu.com/markdown/)
   * [MathJax 特性](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quickreference)
   * [为知笔记 Markdown 新手指南](http://www.wiz.cn/medal-8)
   * [Learning-Markdown](http://xianbai.me/learn-md/index.html)

content

1. 定位
   * 它基于纯文本，方便修改和共享
   * 几乎可以在所有的文本编辑器中编写
   * 有众多编程语言的实现，以及应用的相关扩展
   * 在 GitHub 等网站中有很好的应用
   * 很容易转换为 HTML 文档或其他格式
   * 适合用来编写文档、记录笔记、撰写文章
2. 段落
   * 段落前后必须是空行（空格或制表符）
   * 强制换行使用`<br>`，或者前一行末尾加入两个空格
3. 标题
   * `===`, `#`
   * `----`, `##`
4. 区块引用 blockquotes
   * `>`
   * 整个段落第一行加，或者每一行加
   * 区块引用嵌套
5. 列表
   * 无序号列表：`*`, `+`, `-`
   * 有序号列表：`1.`
   * 如果仅需要在行前显示数字和`.`：使用`1990\.`
6. 分割线: `- - -`
7. 超链接
   * `[Google](http://www.google.com/ 'google' )`
   * `[Google][link]`, `[link]: http://www.google.com/ 'google'`
8. 图像：`![GitHub](https://avatars2.githubusercontent.com/u/3265208?v=3&s=100 "GitHub,Social Coding")`
9. 强调 `**bula**`
10. 演示 `*bula*`
11. 删除线 `~~bula~~`
12. 转义 `\*bula\*`
13. 表格，[生成器](http://jakebathman.github.io/Markdown-Table-Generator/)
14. Task List, `[]`, `[x]`
15. `mathjax` support
    * `\alpht \beta \omega \Gamma \Delta \Omega`
    * `\sum \int \prod \bigcup \bigcap \iint`
    * `\mathbb \Bbb \mathbf \mathtt \mathrm \mathsf \mathcal \mathfrak`
    * `\quad \qquad \bar \overline \hat \widehat \vec \overrightarrow \overleftrightarrow \dot \ddot`
16. history
    * text-to-HTML
    * [Setext](http://docutils.sourceforge.net/mirror/setext.html)
    * [atx](http://www.aaronsw.com/2002/atx/)
    * [Textile](http://textism.com/tools/textile/)
    * [reStructuredText](http://docutils.sourceforge.net/rst.html)
    * [Grutatext](http://www.triptico.com/software/grutatxt.html)
    * [EtText](http://ettext.taint.org/doc/)
17. 定位
    * 适用于网络的书写语言
    * 让文档容易读、写和随意改
    * 只涵盖纯文本可以涵盖的范围（其他通过文档中HTML）
18. 段落与换行：普通段落不该用空格或制表符来缩进

HTML stuff

1. 区块元素
   * 区块标签间的 Markdown 格式语法将不会被处理
   * 部分区块元素必须在前后加上空行与其他内容区隔开，且开始结尾标签不使用制表符和空格：`<div>`, `<table>`, `<pre>`, `<p>`
2. 区段元素
   * `<span>`, `<cite>`, `<del>`, `<a>`, `<img>`
   * Markdown 语法在 HTML 区段标签间是有效的
3. HTML文件中特殊字符转换
   * `<`: 起始标签
   * `&`: 标记 HTML 实体

```html
这是一个普通段落。

<table>
    <tr>
        <td>Foo</td>
    </tr>
</table>

这是另一个普通段落。
```

## Mermaid

1. link
   * [website](https://mermaid.js.org/)
   * [github/mermaid](https://github.com/mermaid-js/mermaid)
   * [vscode-extension/mermaid](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid)

```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```
