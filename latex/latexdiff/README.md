# latexdiff

1. link
   * [CTAN/latexdiff](https://ctan.org/pkg/latexdiff?lang=en)
   * [github/latexdiff](https://github.com/ftilmann/latexdiff/)
   * [overleaf tutorial](https://www.overleaf.com/learn/latex/Articles/Using_Latexdiff_For_Marking_Changes_To_Tex_Documents)
   * [git-latexdiff](https://github.com/rkdarst/git-latexdiff), [gitlab](https://gitlab.com/git-latexdiff/git-latexdiff)

minimum working example (bash/powershell测试通过)

1. `latexdiff draft00.tex draft01.tex > diff.tex`
2. `latexmk -pdf diff.tex`
   * 不保证可以编译通过
   * vscode/latex-workshop没有计划提供该支持
