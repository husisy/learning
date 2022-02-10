'''
reference: https://github.com/explosion/spacy/blob/master/examples/training/train_intent_parser.py

build a message parser for a common "chat intent": finding local businesses
'''

import spacy
import random
from spacy.lang.en import English

train_data = [
    ("find a cafe with great wifi", {
        'heads': [0, 2, 0, 5, 5, 2],
        'deps': ['ROOT', '-', 'PLACE', '-', 'QUALITY', 'ATTRIBUTE']
    }),
    ("find a hotel near the beach", {
        'heads': [0, 2, 0, 5, 5, 2],
        'deps': ['ROOT', '-', 'PLACE', 'QUALITY', '-', 'ATTRIBUTE']
    }),
    ("find me the closest gym that's open late", {
        'heads': [0, 0, 4, 4, 0, 6, 4, 6, 6],
        'deps': ['ROOT', '-', '-', 'QUALITY', 'PLACE', '-', '-', 'ATTRIBUTE', 'TIME']
    }),
    ("show me the cheapest store that sells flowers", {
        'heads': [0, 0, 4, 4, 0, 4, 4, 4],
        'deps': ['ROOT', '-', '-', 'QUALITY', 'PLACE', '-', '-', 'PRODUCT']
    }),
    ("find a nice restaurant in london", {
        'heads': [0, 3, 3, 0, 3, 3],
        'deps': ['ROOT', '-', 'QUALITY', 'PLACE', '-', 'LOCATION']
    }),
    ("show me the coolest hostel in berlin", {
        'heads': [0, 0, 4, 4, 0, 4, 4],
        'deps': ['ROOT', '-', '-', 'QUALITY', 'PLACE', '-', 'LOCATION']
    }),
    ("find a good italian restaurant near work", {
        'heads': [0, 4, 4, 4, 0, 4, 5],
        'deps': ['ROOT', '-', 'QUALITY', 'ATTRIBUTE', 'PLACE', 'ATTRIBUTE', 'LOCATION']
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


doc = nlp_blank('find a hotel with good wifi')
print('Dependencies: ', [(x.text, x.dep_, x.head.text) for x in doc if x.dep_!='-'])
doc = nlp_blank('find me the cheapest gym near work')
print('Dependencies: ', [(x.text, x.dep_, x.head.text) for x in doc if x.dep_!='-'])
doc = nlp_blank('show me the best hotel in berlin')
print('Dependencies: ', [(x.text, x.dep_, x.head.text) for x in doc if x.dep_!='-'])
