import pylatexenc.latexwalker

text_raw = r"""
\textbf{Hi there!} Here is \emph{a list}:
\begin{enumerate}[label=(i)]
\item One
\item Two
\end{enumerate}
and $x$ is a variable.
"""
z0 = pylatexenc.latexwalker.LatexWalker(text_raw.strip())
nodelist, pos, len_ = z0.get_latex_nodes(pos=0)

# pylatexenc.latexwalker.LatexMacroNode
x0 = nodelist[0] #\textbf
x0.macroname #'textbf'
x0.latex_verbatim() #'\textbf{Hi there!}'
x0.nodeargd.argnlist
x0.nodeargs[0].latex_verbatim().lstrip('{').rstrip('}') #'Hi there!'

# pylatexenc.latexwalker.LatexCharsNode
nodelist[1].latex_verbatim() #' Here is '

nodelist[2].nodeargs[0].latex_verbatim().lstrip('{').rstrip('}') #'a list'

nodelist[3] #' \n'

# pylatexenc.latexwalker.LatexEnvironmentNode
x0 = nodelist[4]
x0.environmentname #'enumerate'
x0.nodeargd.argspec #'['
x0.nodeargd.argnlist
x0.nodelist[0] #LatexCharsNode '\n'
x0.nodelist[1] #LatexMacroNode '\item'
x0.nodelist[2] #LatexCharsNode ' One\n'
x0.nodelist[3] #LatexMacroNode '\item'
x0.nodelist[4] #LatexCharsNode ' Two\n'

# pylatexenc.latexwalker.LatexMathNode
nodelist[6].latex_verbatim() #'$x$'
