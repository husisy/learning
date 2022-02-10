import os
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.gold import GoldParse

hf_file = lambda *x: os.path.join('tbd00', *x)


vocab = Vocab(tag_map={'N':{'pos':'NOUN'}, 'V':{'pos':'VERB'}})
doc = Doc(vocab, words=['I', 'like', 'stuff'])
gold = GoldParse(doc, tags=['N', 'V', 'N'])


doc = Doc(Vocab(), words=['Facebook', 'released', 'React', 'in', '2014'])
gold = GoldParse(doc, entities=['U-ORG', 'O', 'U-TECHNOLOGY', 'O', 'U-DATE'])


class SimilarityModel(object):
    def __init__(self, model):
        self._model = model

    def __call__(self, doc):
        doc.user_hooks['similarity'] = self.similarity
        doc.user_span_hooks['similarity'] = self.similarity
        doc.user_token_hooks['similarity'] = self.similarity

    def similarity(self, obj1, obj2):
        y = self._model([obj1.vector, obj2.vector])
        return float(y[0])


nlp_sm = spacy.load('en_core_web_sm')
doc1 = nlp_sm('Apple is looking at buying U.K. startup for $1 billion')
doc1.to_disk(hf_file('test01.bin'))
doc2 = spacy.tokens.Doc(spacy.vocab.Vocab()).from_disk(hf_file('test01.bin'))
assert all(x.text==y.text for x,y in zip(doc1,doc2))
