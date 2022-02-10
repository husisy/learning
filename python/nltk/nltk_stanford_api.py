from nltk.corpus import treebank
from nltk.parse import DependencyGraph
from nltk.parse import CoreNLPParser, CoreNLPDependencyParser

parser = CoreNLPParser('http://localhost:9000')
dep_parser = CoreNLPDependencyParser('http://localhost:9000', tagtype='pos')
pos_tagger = CoreNLPParser('http://localhost:9000', tagtype='pos')
ner_tagger = CoreNLPParser('http://localhost:9000', tagtype='ner')
# iterator, when possible, is sorted from most likely to least likely. Pick the first one here

raw = 'What is the airspeed of an unladen swallow?'
raw_sent = 'John loves Mary. Mary walks.'
token = 'What is the airspeed of an unladen swallow ?'.split(' ')
token1 = 'Rami Eid is studying at Stony Brook University in NY'.split()
raw_list = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick grey wolf jumps over the lazy fox.',
    "This is my friends' cat (the tabby)",
]

# Tokenizer
list(parser.tokenize(raw))

# ConstituentTree
next(parser.parse(token))
next(parser.raw_parse(raw))
[next(x) for x in parser.raw_parse_sents(raw_list)]
list(parser.parse_text(raw_sent))

# DependencyTree
hf_deptree_to_bracketstr = lambda dg: str(dg.tree()) #NO POS (not possible)
hf_deptree_to_conll = lambda dg: [(x['word'],x['tag'],x['head'],x['rel']) for _,x in sorted(dg.nodes.items()) if x['tag']!='TOP']
hf_conll_to_deptree = lambda conll: DependencyGraph('\n'.join('\t'.join([x1,x2,str(x3),x4]) for x1,x2,x3,x4 in conll))

next(dep_parser.raw_parse(raw))
z1 = next(dep_parser.parse(token))
list(z1.triples())

# POS
pos_tagger.tag(token)
[x[0] for x in pos_tagger.raw_tag_sents(raw_list)]
pos_tagger.tag_sents

# NER
ner_tagger.tag(token1)


# z1 = list(treebank.sents())
# z2 = pos_tagger.tag_sents(z1)
# pos_tagger_set = set(y[1] for x in z2 for y in x)

# z3 = dep_parser.tag_sents(z1)
# dep_parser_set = set(y[1] for x in z3 for y in x)

# z4 = [next(dep_parser.parse(x)) for x in z1]
# dep_parser_set1 = {y['tag']  for x in z4 for _,y in x.nodes.items() if y['ctag']!='TOP'}