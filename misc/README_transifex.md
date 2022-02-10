# transifex

这个文档将通过翻译python-doc的方式来介绍transifex使用，以及其他用户该如何参与到其中。
为个人方便，文档初期依旧以个人习惯的方式记录（例如中英文混合、个人习惯的缩写等），之后在做进一步的整理(TODO)。

## overview

1. [official site](https://www.transifex.com/)
2. [documentation](https://docs.transifex.com/)
   * [doc0](https://docs.transifex.com/getting-started-1/translators)
   * [doc1](https://docs.transifex.com/translation/translating-with-the-web-editor)
3. account (social login only): github, gmail
4. term
   * string
5. TODO
   * github-大陆简中自由软件本地化工作指南
   * transifex doc

## Python-doc Chinese

1. [python-doc翻译项目公开链接](https://www.transifex.com/python-doc/public/)
   * organization: Python document translations
   * project: Python
   * team: Chinese (China)
2. 参与方法：见[transifex-doc](https://docs.transifex.com/getting-started-1/translators#joining-a-translation-team)
   * 创建用户：[transifex-website](https://www.transifex.com/)
   * 打开[python-doc翻译项目公开链接](https://www.transifex.com/python-doc/public/)
   * 点击`join team`
   * 选择`Chinese (China)`
   * 点击`Join`
   * 等待通过(约1小时，会通过邮件告知)
3. 参考资源。下文中的断言如无特殊说明，则参考自如下的链接内容
   * [PEP545](https://www.python.org/dev/peps/pep-0545/)
   * [术语对照表](https://docs.python.org/zh-cn/3/glossary.html)
   * [翻译进度跟踪以及常见问题汇总](https://github.com/python/python-docs-zh-cn/issues/10)
   * [大陆简中自由软件本地化工作指南](http://mirrors.ustc.edu.cn/anthon/aosc-l10n/zh_CN_l10n.pdf)
   * [文档错误反馈](https://docs.python.org/zh-cn/3/bugs.html#documentation-bugs)
   * [简书-Python官方文档中文翻译项目](https://www.jianshu.com/p/27d2f02a86e9)
   * [github-repo-wiki](https://github.com/python/python-docs-zh-cn/wiki)
   * [github-python-docs-zh-cn](https://github.com/python/python-docs-zh-cn)
   * [招募翻译review人员](https://github.com/python/python-docs-zh-cn/issues/1)
   * [翻译任务协调](https://github.com/python/python-docs-zh-cn/issues/6)
4. 项目免责声明（以下近乎为引用，如有冒犯，不许打我，手动滑稽）
   * 暂无分工安排，完全根据自己时间和爱好，单独翻译某一篇、某一个段，甚至是单独的一行或短句也行
   * 管理员很懒的，没有公告和手册，很多都要靠自己摸索
   * 看样子除了github-repo，暂无qq群、微信群、slack等其他交流平台
5. 所有的翻译工作都在[python-doc翻译项目公开链接](https://www.transifex.com/python-doc/public/)中完成(而非git repo)
6. 翻译版文档会定时更新到[官网](https://docs.python.org/zh-cn/3/)以便对照参阅
7. [review](https://github.com/python/python-docs-zh-cn/issues/1)
8. 如已加入transifex-team，一个较好的翻译参考示例`library-turtle`
9. 对于[术语对照表](https://docs.python.org/zh-cn/3/glossary.html)中内容，保持翻译一致性
10. transifex的web编辑器保存后，他人能马上看得到，应该不会出现重复工作
11. 常见问题
    * `rst`格式标记的前后应有分隔符，如空格、逗号、句号等(。。。啥是rst格式标记)
    * markdown-escape-string00: 保持原样不改动内容；即**不**将`:`改为`：`，**不**翻译`xxx`
    * markdown-escape-string01: 内容可翻译，建议先复制源文本区到翻译区，再翻译`xxx`
    * markdown-escape-string02：内容可翻译
    * markdown-escape-string03：内容可翻译

```python
# markdown-escape-string
# 00%:term:`xxx`%
# 01%:term:`xxx (1]`%
# 02%*xxx*%
# 03%``xxx``%
```
