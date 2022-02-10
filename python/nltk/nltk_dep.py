from collections import defaultdict

import nltk
from nltk.tree import Tree
from nltk.corpus import treebank, ppattach
from nltk.parse import DependencyGraph


# valency
# complement
# modifier

# Lexical Function Grammar (LFG) Pargram project
# Head-Driven Phrase Structure Grammar (HPSG)
# LinGO Matrix framework
# Lexicalized Tree Adjoining Grammar XTAG Project


tmp1 = [
    "'shot' -> 'I' | 'elephant' | 'in'",
    "'elephant' -> 'an' | 'in'",
    "'in' -> 'pajamas'",
    "'pajamas' -> 'my'",
]
grammar = nltk.DependencyGrammar.fromstring('\n'.join(tmp1))
token = ['I', 'shot', 'elephant', 'in', 'my', 'pajamas']
parser = nltk.ProjectiveDependencyParser(grammar)
z1 = list(parser.parse(token))


z1 = treebank.parsed_sents('wsj_0001.mrg') #still CFG

hf1 = lambda t: (t.label()=='VP') and any('S'==x.label() for x in t if isinstance(x, Tree))
hf1 = lambda t: (t.label()=='VP') and len(t)>1 and isinstance(t[1],Tree) and t[1].label()=='S'
z1 = [t for x in treebank.parsed_sents() for t in x.subtrees(hf1)]


# ppattach: verb noun1 preposition noun2
entries = ppattach.attachments('training')
table = defaultdict(lambda: defaultdict(set))
for entry in entries:
    key = entry.noun1 + '-' + entry.prep + '-' + entry.noun2
    table[key][entry.attachment].add(entry.verb)
count = 0
for key in sorted(table):
    if len(table[key]) > 1:
        print(key, 'N:', sorted(table[key]['N']), 'V:', sorted(table[key]['V']))
        count += 1
    if count>100: break


treebank_data = """Pierre  NNP     2       NMOD
Vinken  NNP     8       SUB
,       ,       2       P
61      CD      5       NMOD
years   NNS     6       AMOD
old     JJ      2       NMOD
,       ,       2       P
will    MD      0       ROOT
join    VB      8       VC
the     DT      11      NMOD
board   NN      9       OBJ
as      IN      9       VMOD
a       DT      15      NMOD
nonexecutive    JJ      15      NMOD
director        NN      12      PMOD
Nov.    NNP     9       VMOD
29      CD      16      NMOD
.       .       9       VMOD
"""
conll_data = """
1   Ze                ze                Pron  Pron  per|3|evofmv|nom                 2   su      _  _
2   had               heb               V     V     trans|ovt|1of2of3|ev             0   ROOT    _  _
3   met               met               Prep  Prep  voor                             8   mod     _  _
4   haar              haar              Pron  Pron  bez|3|ev|neut|attr               5   det     _  _
5   moeder            moeder            N     N     soort|ev|neut                    3   obj1    _  _
6   kunnen            kan               V     V     hulp|ott|1of2of3|mv              2   vc      _  _
7   gaan              ga                V     V     hulp|inf                         6   vc      _  _
8   winkelen          winkel            V     V     intrans|inf                      11  cnj     _  _
9   ,                 ,                 Punc  Punc  komma                            8   punct   _  _
10  zwemmen           zwem              V     V     intrans|inf                      11  cnj     _  _
11  of                of                Conj  Conj  neven                            7   vc      _  _
12  terrassen         terras            N     N     soort|mv|neut                    11  cnj     _  _
13  .                 .                 Punc  Punc  punt                             12  punct   _  _
"""
dg = DependencyGraph(conll_data)
tree = dg.tree()
print(dg)
print(dg.to_conll(4))

dg = DependencyGraph(treebank_data)
[(x['word'],x['tag'],x['head'],x['rel']) for _,x in sorted(dg.nodes.items()) if x['tag']!='TOP']
