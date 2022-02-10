import spacy

nlp_sm = spacy.load('en_core_web_sm')
# nlp_md = spacy.load('en_core_web_md')


# string hash
doc = nlp_sm('I love coffee')
coffee_hash = nlp_sm.vocab.strings['coffee']
assert coffee_hash==doc[2].orth #int
assert nlp_sm.vocab.strings[coffee_hash]==doc[2].text #str


# doc basic attributes
doc = nlp_sm('Apple is looking at buying U.K. startup for $1 billion')
str_fmt = '{:>10} {:>3} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'
print(str_fmt.format('text', 'i', 'lemma_', 'pos_', 'tag_', 'dep_', 'shape_', 'is_alpha', 'is_stop', 'is_punct'))
for x in doc:
    print(str_fmt.format(x.text, x.i, x.lemma_, x.pos_, x.tag_, x.dep_, x.shape_, x.is_alpha, x.is_stop, x.is_punct))
str_fmt = '{:>10} {:>10} {:>22} {:>22} {:>22}'

print('')
print(str_fmt.format('text', 'pos', 'tag', 'dep', 'shape'))
for x in doc:
    print(str_fmt.format(x.text, x.pos, x.tag, x.dep, x.shape))


# ner attributes
doc = nlp_sm('Apple is looking at buying U.K. startup for $1 billion')
str_fmt = '{:>10} {:>12} {:>10} {:>10}'
print(str_fmt.format('text', 'start_char', 'end_char', 'label_'))
for x in doc.ents:
    print(str_fmt.format(x.text, x.start_char, x.end_char, x.label_))

print('')
str_fmt = '{:>10} {:>10} {:>10}'
print(str_fmt.format('text', 'ent_iob_', 'ent_type'))
for x in doc:
    print(str_fmt.format(x.text, x.ent_iob_, x.ent_type))

print('')
if input('type YES to confirm displacy.serve(ent): ')=='YES':
    spacy.displacy.serve(doc, style='ent')


# noun chunks
doc = nlp_sm('Autonomous cars shift insurance liability toward manufacturers')
str_fmt = '{:>20} {:>15} {:>15} {:>15}'
print(str_fmt.format('text', 'root.text', 'root.dep_', 'root.head.text'))
for x in doc.noun_chunks:
    print(str_fmt.format(x.text, x.root.text, x.root.dep_, x.root.head.text))


# pick token
doc = nlp_sm('Autonomous cars shift insurance liability toward manufacturers')
hf1 = lambda x: x.dep==spacy.symbols.nsubj and x.head.pos==spacy.symbols.VERB
print('dep==nsubj and head.pos==VERB: ', [x.text for x in doc if hf1(x)])


# span merge
nlp01 = spacy.lang.en.English()
doc = nlp01('a aa aaa aaaa aaaaa')
print('doc.text: ', [x.text for x in doc])
doc[1:3].merge()
print('doc.text: ', [x.text for x in doc])


# export to numpy array
doc = nlp_sm('Check out https://spacy.io')
for x in doc: print(x.text, x.orth, x.like_url)
doc_array = doc.to_array([spacy.attrs.ORTH, spacy.attrs.LIKE_URL])
print(doc_array)
