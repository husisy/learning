import re
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH, LEMMA, POS, TAG

nlp_sm = spacy.load('en_core_web_sm')


# speicial case
text = 'gimme that'
print([x.text for x in nlp_sm(text)])

tmp1 = [{ORTH:'gim', LEMMA:'give', POS:'VERB'}, {ORTH:'me'}]
nlp_sm.tokenizer.add_special_case('gimme', tmp1)
doc = nlp_sm(text)
print('after text:  ', [x.text for x in doc])
print('after lemma_:', [x.lemma_ for x in doc])

nlp_sm = spacy.load('en_core_web_sm')


#customize tokenizer
text = 'hello-world.'
print([x.text for x in nlp_sm(text)])

tmp1 = re.compile(r'''^[\[\("']''')
tmp2 = re.compile(r'''[\]\)"']$''')
tmp3 = re.compile(r'''[-~]''')
simple_url_re = re.compile(r'''^https?://''')
nlp_sm.tokenizer = Tokenizer(nlp_sm.vocab, prefix_search=tmp1.search,
        suffix_search=tmp2.search, infix_finditer=tmp3.finditer, token_match=simple_url_re.match)
print([x.text for x in nlp_sm(text)])

nlp_sm = spacy.load('en_core_web_sm')


# whitespace tokenizer
text = "What's happened to me? he thought. It wasn't a dream."
print('before: ', [x.text for x in nlp_sm(text)])
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
    def __call__(self, text):
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp_sm.tokenizer = WhitespaceTokenizer(nlp_sm.vocab)
print('after: ', [x.text for x in nlp_sm(text)])

nlp_sm = spacy.load('en_core_web_sm')


# whitespace
nlp_english = English()
doc = Doc(nlp_english.vocab, words=['Hello',',','world','!'], spaces=[False,True,False,False])
print([(x.text,x.text_with_ws,x.whitespace_) for x in doc])
doc = nlp_sm('Hello, world!')
print([(x.text,x.text_with_ws,x.whitespace_) for x in doc])
