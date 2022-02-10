import os
import spacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher


# from utils import next_tbd_dir

# logdir = next_tbd_dir()
# hf_file = lambda *x,dir0=logdir: os.path.join(dir0, *x)

nlp_sm = spacy.load('en_core_web_sm')


# custom pipeline components
def my_component(doc):
    print('after tokenization, this doc has {} tokens.'.format(len(doc)))
    return doc
nlp_sm.add_pipe(my_component, name='print_info', first=True)
print(nlp_sm.pipe_names)
doc = nlp_sm('This is a sentence')

nlp_sm.remove_pipe('print_info')


# custom pipeline components
class EntityMatcher(object):
    name = 'entity_matcher'
    def __init__(self, nlp, terms, label):
        patterns = [nlp(text) for text in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add(label, None, *patterns)
    def __call__(self, doc):
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = Span(doc, start, end, label=match_id)
            doc.ents = list(doc.ents) + [span]
        return doc
terms = ('cat', 'dog', 'tree kangaroo', 'giant sea spider')
entity_matcher = EntityMatcher(nlp_sm, terms, 'ANIMAL')
nlp_sm.add_pipe(entity_matcher, after='ner')
print(nlp_sm.pipe_names)
doc = nlp_sm('This is a text about Barack Obama and a tree kangaroo')
print([(ent.text, ent.label_) for ent in doc.ents])

nlp_sm.remove_pipe(entity_matcher.name)
