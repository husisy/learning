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


#
nlp_md = spacy.load('en_core_web_md')
with nlp_md.disable_pipes(*[pipe for pipe in nlp_md.pipe_names if pipe!='ner']): #only train ner
    optimizer  = nlp_md.begin_training()
    for ind1 in range(10):
        random.shuffle(train_data)
        for text, annotation in train_data:
            nlp_md.update([text], [annotation], sgd=optimizer)
nlp_md.to_disk(next_tbd_dir())

nlp_md = spacy.load('en_core_web_md')


