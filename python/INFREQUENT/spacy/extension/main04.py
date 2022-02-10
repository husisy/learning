import os
import spacy
from spacy.tokens import Doc, Span, Token

nlp_sm = spacy.load('en_core_web_sm')

#
fruits = ['apple', 'pear', 'banana', 'orange', 'strawberry']
is_fruit_getter = lambda token: token.text in fruits
has_fruit_getter = lambda x: any((t.text in fruits) for t in x)

Token.set_extension('is_fruit', getter=is_fruit_getter)
Doc.set_extension('has_fruit', getter=has_fruit_getter)
Span.set_extension('has_fruit', getter=has_fruit_getter)
doc = nlp_sm('apple and pear and banana and orange')
print([x.text for x in doc])
print([x._.is_fruit for x in doc])
print(doc._.has_fruit)
Token.remove_extension('is_fruit')
Doc.remove_extension('has_fruit')
Span.remove_extension('has_fruit')


# custom sentence segmentation logic
def sbd_component(doc):
    for i,token in enumerate(doc[:-2]):
        if token.text=='.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
    return doc
nlp_sm.add_pipe(sbd_component, before='parser')
doc = nlp_sm('This is a sentence. This is another sentence.')
for sent in doc.sents:
    print(sent)

nlp_sm.remove_pipe(sbd_component.__name__)

