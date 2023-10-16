# multi language support

1. link
   * [overleaf-link](https://www.overleaf.com/learn/latex/International_language_support) International language support
   * [overleaf-link](https://www.overleaf.com/learn/latex/International_language_support#Reference_guide) accents and special character
   * [wikibook-link](https://en.wikibooks.org/wiki/LaTeX/Special_Characters) LaTeX/Special Characters
2. package `babel`
   * change "abstract" to the spanish words "resumen"
3. texlive-2018 adopt `utf-8` as default text encoding
   * no need `\usepackage[utf8]{inputenc}`
4. font encoding `\usepackage[encoding]{fontenc}`
   * `OT1`
   * `T1`
   * `T2A`
   * `T2B`
   * `T2C`
   * `X2`
5. switch pdfLaTeX to LuaLaTeX or XeLaTeX
