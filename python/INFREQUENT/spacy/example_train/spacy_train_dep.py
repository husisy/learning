'''
reference: https://github.com/explosion/spacy/blob/master/examples/training/train_parser.py

train spaCy dependency parser, starting off with a blank model
'''

import spacy
import random

train_data = [
    ("They trade mortgage-backed securities.", {
        'heads': [1, 1, 4, 4, 5, 1, 1],
        'deps': ['nsubj', 'ROOT', 'compound', 'punct', 'nmod', 'dobj', 'punct']
    }),
    ("I like London and Berlin.", {
        'heads': [1, 1, 1, 2, 2, 1],
        'deps': ['nsubj', 'ROOT', 'dobj', 'cc', 'conj', 'punct']
    })
]


nlp_blank = spacy.blank('en')
parser = nlp_blank.create_pipe('parser')
nlp_blank.add_pipe(parser, first=True)

label = set(y for x in train_data for y in x[1].get('deps'))
for x in label:
    parser.add_label(x)

optimizer = nlp_blank.begin_training()
for _ in range(10):
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        nlp_blank.update([text], [annotations], sgd=optimizer, losses=losses)
    print(losses)

doc = nlp_blank('I like securities.')
print('Dependencies: ', [(t.text, t.dep_, t.head.text) for t in doc])
