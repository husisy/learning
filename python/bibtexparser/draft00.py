import bibtexparser

bibtex_str = """
@comment{
    This is my example comment.
}

@ARTICLE{Cesar2013,
  author = {Jean César},
  title = {An amazing title},
  year = {2013},
  volume = {12},
  pages = {12--23},
  journal = {Nice Journal}
}
"""
# bibtexparser.parse_file("my_file.bib")
library = bibtexparser.parse_string(bibtex_str)
len(library.blocks) #2
len(library.entries) #1
len(library.comments) #1
len(library.strings) #0
len(library.preambles) #0

x0 = library.comments[0]
x0.comment #str

x1 = library.entries[0]
x1.key #"Cesar2013"
x1.entry_type #article
x1.fields #list
x1.fields_dict #list
x1f = x1.fields[0]
x1f.key #author
x1f.value #"Jean César"
