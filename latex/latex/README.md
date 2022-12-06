# LaTeX

**NOTICE**: 当前文档，所有backslash `\` 都替换为slash `/`

TODO

1. [x] [latexdiff](https://www.overleaf.com/learn/latex/Articles/Using_Latexdiff_For_Marking_Changes_To_Tex_Documents)
2. [ ] lyx: introduction and tutorial in software

LaTeX世界观

1. link
   * [official site](https://www.tug.org/)
   * [一份其实很短的 LaTeX 入门文档](https://liam.page/2014/09/08/latex-introduction/)
   * [手动编译](https://en.wikibooks.org/wiki/LaTeX/Basics#Compilation)
   * [texlive-zh-cn.pdf](http://www.tug.org/texlive/doc/texlive-zh-cn/texlive-zh-cn.pdf)
   * [github/Ctex-org](https://github.com/CTeX-org)
   * [CTAN/documentclass](https://ctan.org/topic/class)
   * [oeis/latex-math-symbols](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)
2. 安装说明
   * 官网上推荐win用户安装proTeXt，*未测试*，以下文字都基于TeXLive，see [link](https://www.tug.org/begin.html)
   * [下载链接的网站](https://www.tug.org/texlive/acquire-netinstall.html)，[install-tl-windows.exe](http://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe)，[install-tl-unx.tar.gz](http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz)
   * [quick install for unix](https://www.tug.org/texlive/quickinstall.html)，**不需要**管理员权限，虽然也可以使用包管理器安装
   * [quick install for windows](https://www.tug.org/texlive/windows.html)，几乎就是鼠标一路点到底
3. 幺蛾子名词：`TeX`, `LaTeX`, `pdfTeX`, `pdfLaTeX`, `XeTeX`, `XeLaTeX`, `LuaTeX`
4. TeX发行（TeX发行版、TeX系统、TeX套装）：指包括TeX系统的各种可执行程序，以及他们执行时需要的一些辅助程序和宏包文档的集合: `CTeX`, `MiKTeX`, `TeXLive`
5. 换行符模式：`CRLF`(win), `LF`(*nix)
6. 转义字符escaped character: `/#%$^&_{}~`
7. 偏见
   * 使用xelatex而非pdflatex：减少中英文排版思考成本，减少.tex文件编码成本（默认utf8），增加编译时间
   * 使用CTeX宏包（即`ctexart`等），**禁止**使用CJK宏包
8. [choosing a latex compiler](https://www.overleaf.com/learn/latex/Choosing%20a%20LaTeX%20Compiler)
   * tex typesetting / tex distribution: `MiKTeX`, `TeXLive`, `MacTeX`
   * tex to dvi: `latex xxx.tex`, device independent file format
   * tex to pdf: `pdflatex xxx.tex`, portable Document format
   * dvi to pdf: `dvipdfmx`
   * dvi to ps: `dvips`, PostScript file format
   * ps to pdf: `ps2pdf`
   * LaTeX compiler: only support `.eps` `.ps` image
   * pdfLaTeX: support `.png`, `.jpg`, `.pdf`
   * XeLaTeX, LuaLaTeX: support UTF8
9. vscode-latex-workshop
   * `ctrl+alt+v`
   * `ctrl+click`
   * `ctrl+alt+j`
10. documentclass [tex-stackexchange](https://tex.stackexchange.com/q/782)
    * `article`: scientific journals, presentations, short reports, program documentation, invitations
    * `proc`: for proceedings based on the article class
    * `minimal` for debugging purposes
    * `report`: small books, thesis
    * `book`
    * `slides`: for slides. The class uses big sans serif letters
    * `letter`
    * `beamer`: presentations

```bash
tlmgr init-usertree
tlmgr option repository ftp://tug.org/historic/systems/texlive/2017/tlnet-final
tlmgr option repository http://mirror.ctan.org/systems/texlive/tlnet
tlmgr install cctbook
update-tlmgr-latest
tlmgr option repository https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet
tlmgr remove --all

# wget https://mirror.ctan.org/systems/texlive/tlnet/update-tlmgr-latest.sh
# bash ./update-tlmgr-latest.sh
```

杂项

1. link
   * [一份其实很短的 LaTeX 入门文档](https://liam.page/2014/09/08/latex-introduction/)
   * [ZIP 归档](https://liam.page/attachment/attachment/LaTeX-useful-tools/LaTeX_Docs_2014.zip)
   * [Overleaf - Learn LaTeX in 30 minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)
   * [The Not So Short Introduction To latex (Chinese Edition)](https://ctan.org/tex-archive/info/lshort/chinese?lang=en)
   * [官网上的quickstart](https://www.tug.org/begin.html)
2. 偏见
   * 使用`$...$`而非使用`/(.../)`，而非使用`/begin{math}.../end{math}`
   * 使用`align`环境而非用`eqnarray`环境
   * 使用`matrix bmatrix pmatrix vmatrix Vmatrix`等环境而非`array`环境去实现矩阵
   * 使用`/[.../]`而非使用`$$...$$`，而非使用`/begin{displaymath}.../end{displaymath}`，而非`/begin{equation*}.../end{equation*}`，见[孟晨-知乎](https://www.zhihu.com/question/27589739/answer/37237684)
   * 中文破折号`shift -`（中文输入法下） [link0](https://www.zhihu.com/question/338442037/answer/776619663) [link1](https://www.thetype.com/2019/03/14918/)
3. 定界符
   * `() [] /{/} /langle/rangle /lvert/rvert /lVert/rvert`
   * 从小到大依次嵌套：`. /big /Big /bigg /Bigg`
   * 使用`/bigl.../bigr`而非`/left.../right`，见[孟晨-知乎](https://www.zhihu.com/question/27598898/answer/37276221)
   * 使用`/lvert/rvert /lVert/rVert`而非`| \|`
4. `/dots`相比`/cdots`，前者一般用于有下标的序列
5. basic levels of depth
   * `/part{}`: -1, only available in report and book
   * `/chapter{}`: 0, only available in report and book
   * `/section{}`: 1
   * `/subsection{}`: 2
   * `/subsubsection{}`: 3
   * `/paragraph{}`: 4
   * `/subparagraph{}`: 5
6. 浮动体`htbp`, here, top, bottom, float page（专门放浮动体的单独页面或分栏）
   * `/label`放在`/caption`之后
   * 见[孟晨-知乎](https://www.zhihu.com/question/25082703/answer/30038248)
7. 版面设置：页边距`geometry`，页眉页脚`fancyhdr`，行间距`setspace`
8. table generator tool: [website](https://www.tablesgenerator.com/)

## lyx

1. link
   * [official-site](https://www.lyx.org/)
   * [lyx-wiki](https://wiki.lyx.org/)
   * [tutorial](https://wiki.lyx.org/LyX/Tutorials)
   * `lyx-software/Help/introduction`
   * `lyx-software/Help/tutorial`
   * `lyx-software/Help/UserGuide`
   * `lyx-software/examples/*.lyx`
2. lyx中用户被禁止的操作
   * tab stops
   * extra whitespace
   * indenting control
   * line spacing
   * whitespace, horizontal and vertical
   * fonts and font size
   * typeface (bold, italic, underline)
3. shortcut
   * `ctrl+R`: Document, view pdf
4. 关于空格
   * `Return/Enter`: separate paragraph
   * `Space` separate word
   * `Tab`: no meaning
   * protected space: `ctrl+shift+space`
5. what you see is what you mean
6. 常用指令
   * 设置environment (title, author, date, section, quote, lyx-code, verse), quote短引用，quotation长引用
   * 插入TOC: `Insert/ListTOC/TOC`, outline panel
   * 列表
   * footnote, margin note: `Insert/Footnote`
   * cross reference: `Insert/Label`, `Insert/CrossReference`, protected space, copy label
   * documentclass: `Document/Settings`
   * bibliograph: `Environment/Bibliography`, `Insert/Citation`
   * math: `ctrl+m` inlined formula, `ctrl+shift+m` displayed formula
7. 环境层次：section, subsection, subsubsection paragraph, subparagraph (indented)
8. 列表
   * slide使用`itemize`
   * outline使用`enumerate`
   * document describing several software packages could use `descritpion` environment where each item in the list begins with a bold-faced word
9. documentclass
   * `article`: one-sided, no chapter
   * `article(AMS)`
   * `report`: longer than article, two-sided
   * `book`: report, front and back matter

### mwe00

1. `ctrl+N`创建新文件
2. 输入`this is title`，点击工具栏environment（File的下一行，有下拉列表，默认是Standard的那个块块），选择`Title`
3. 换行，输入`this is content`
4. `ctrl+R`进行`pdflatex`编译并预览pdf文件

## advice for writing latex

1. link
   * [github - latex-advices](https://github.com/dspinellis/latex-advice)
   * [LaTeX Tips: The Top Ten List](https://faculty.math.illinois.edu/~hildebr/tex/tips-topten.html)
2. avoid long lines, start each phrase on a separate line
3. use `$TODO$` equation tag
4. automate the document build: vscode-latex-workshop
5. continuous integration: *TODO*, travis-ci
6. bibliographics references management
   * prefer Biber to BibTeX
   * `/cite`, `/citet`, `/citep`, `/citeauthor`
   * Endnote
   * consistent short names for bibliography: `Qiao2008`, `Qiao2008a`, `ABCD2009`
7. third-party latex packages
   * [algorithmicx](https://www.ctan.org/pkg/algorithmicx): Display good-looking pseudocode
   * [amsmath, amssymb](https://www.ctan.org/pkg/amsmath): AMS mathematical facilities
   * [amsthm](https://www.ctan.org/pkg/amsthm): Typesetting theorems (AMS style)
   * [booktabs](https://www.ctan.org/pkg/booktabs): Publication quality tables
   * [cite](https://www.ctan.org/pkg/cite): Improved citation handling
   * [fancyhdr](https://www.ctan.org/pkg/fancyhdr): Extensive control of page headers and footers
   * [geometry](https://www.ctan.org/pkg/geometry): Flexible and complete interface to document dimensions
   * [hyperref](https://www.ctan.org/pkg/hyperref): Extensive support for hypertext
   * [listings](https://www.ctan.org/pkg/listings): Typeset source code listings
   * [minted](https://www.ctan.org/pkg/minted): Typeset source code listings with highligthing
   * [natbib](https://www.ctan.org/pkg/natbib): Flexible bibliography support
   * [PGF/TikZ](https://www.ctan.org/pkg/pgf): Create PostScript and PDF graphics
   * [setspace](https://www.ctan.org/pkg/setspace): Set space between lines
   * [siunitx](http://www.ctan.org/pkg/siunitx): A comprehensive (SI) units package
   * [url](https://www.ctan.org/pkg/url): Verbatim with URL-sensitive line breaks
   * [xcolor](https://www.ctan.org/pkg/xcolor): Driver-independent color extensions
   * [xspace](https://www.ctan.org/pkg/xspace): Define commands that appear not to eat spaces
   * [cleveref](https://www.ctan.org/pkg/cleveref): Intelligent cross-referencing
8. use `/emph` rather `/textit`
9. multi-character variable name use `/mathit{Delta}` or `/mathrm{Delta}`

## minimum working example

纯英文文档 `draft00.tex`

```latex
\documentclass{article}
\title{xxx-Title}
\author{xxx-Author}
\date{\today}
\begin{document}
\maketitle
scene-00\\
user: hello word\\
latex: mmp
\end{document}
```

1. 编译方式一
   * `pdflatex draft00.tex`
   * 可选参数`-output-directory=tbd00`，需先创建`tbd00`目录
2. 编译方式二
   * `latexmk -pdf draft00.tex`
   * 清理文件 `latexmk -c`

中英文混合文档 `draft00.tex`

```latex
\documentclass[UTF8]{ctexart}
\title{xxx-Title}
\author{xxx-Author}
\date{\today}
\begin{document}
\maketitle
脑补小剧场\\
user: 你好 word\\
latex: mmp
\end{document}
```

1. 编译方式一
   * `xelatex draft00.tex`
   * 可选参数`-output-directory=tbd00`，需先创建`tbd00`目录
2. 编译方式二
   * `latexmk -pdf draft00.tex`
   * 清理文件 `latexmk -c`
3. 在ubuntu平台编译失败, see [github-issue-xetex中fandol字体script的问题](https://github.com/CTeX-org/forum/issues/34)
   * `/documentclass[UTF8,fontset=ubuntu]{ctexart}`，然后使用xelatex编译通过，但vscode remote ssh下未通过
   * see [link](https://stackoverflow.com/a/57734531/7290857)
   * see [link](https://tex.stackexchange.com/a/284933)
   * 但xelatex不解决引用的问题，整体上说windows下使用latexmk就足够了，但暂未测试出跨平台兼容的解决方案，尤其时remote ssh下问题较多

## ws-bib

1. link
   * [overleaf/Bibliography management with bibtex](https://www.overleaf.com/learn/latex/bibliography_management_with_bibtex)
