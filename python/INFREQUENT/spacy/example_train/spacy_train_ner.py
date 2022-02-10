'''
reference: https://github.com/explosion/spacy/blob/master/examples/training/train_ner.py

train spaCy's named entity recognizer starting off with a blank model
'''
import os
import spacy
import random

from utils import next_tbd_dir

logdir = next_tbd_dir()

train_data = [
    ('Uber blew through $1 million a week', {'entities':[(0,4,'ORG')]}),
    ('Google rebrands its business apps', {'entities':[(0,6,'ORG')]}),
    ('Who is Shaka Khan?', {'entities':[(7,17,'PERSON')]}),
    ('I like London and Berlin.', {'entities':[(7,13,'LOC'), (18,24,'LOC')]})
]


nlp_blank = spacy.blank('en')
ner = nlp_blank.create_pipe('ner')
nlp_blank.add_pipe(ner, last=True)

label = set(y[2] for x in train_data for y in x[1].get('entities'))
for x in label:
    ner.add_label(x)

optimizer = nlp_blank.begin_training()
for _ in range(50):
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        nlp_blank.update( [text], [annotations], drop=0.5, sgd=optimizer, losses=losses)
    print(losses)

for text, label in train_data:
    doc = nlp_blank(text)
    print('Entities: ', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens:   ', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    print('gt label: ', label.get('entities'))
    print('')

nlp_blank.to_disk(logdir)

nlp_blank_ = spacy.load(logdir)
for text, label in train_data:
    doc = nlp_blank_(text)
    print('Entities: ', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens:   ', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    print('gt label: ', label.get('entities'))
    print('')
