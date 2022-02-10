import nltk


kim = {'CAT':'NP', 'ORTH':'Kim', 'REF':'k'}
chase = {'CAT':'V', 'ORTH':'chased', 'REL':'chase'}
lee = {'CAT':'NP', 'ORTH':'Lee', 'REF':'l'}

token = ['Kim', 'chased', 'Lee']
subj, verb, obj = kim, chase, lee
verb['AGT'] = subj['REF']
verb['PAT'] = obj['REF']
for k in ['ORTH', 'REL', 'AGT', 'PAT']:
    print('{:5} => {}'.format(k, verb[k]))


nltk.data.show_cfg('grammars/book_grammars/feat0.fcfg')
grammar = nltk.data.load('grammars/book_grammars/feat0.fcfg')
parser = nltk.load_parser('grammars/book_grammars/feat0.fcfg', trace=2)
token = ['Kim', 'likes', 'children']
z1 = list(parser.parse(token))


# attribute value matrix
fs1 = nltk.FeatStruct(TENSE='past', NUM='sg', GND='fem')
fs1['GND']
fs1['CASE'] = 'acc'
fs2 = nltk.FeatStruct(POS='N', AGR=fs1)