'''
reference: https://github.com/explosion/spacy/blob/master/examples/training/train_tagger.py
universal tag set: http://universaldependencies.github.io/docs/u/pos/index.html

you may need specify morphological features for tags from the universal scheme
'''
import spacy
import random

TAG_MAP = {
    'N': {'pos': 'NOUN'},
    'V': {'pos': 'VERB'},
    'J': {'pos': 'ADJ'}
}

train_data = [
    ("I like green eggs", {'tags': ['N', 'V', 'J', 'N']}),
    ("Eat blue ham", {'tags': ['V', 'J', 'N']})
]

nlp_blank = spacy.blank('en')
tagger = nlp_blank.create_pipe('tagger')
for tag,value in TAG_MAP.items():
    tagger.add_label(tag, value)
nlp_blank.add_pipe(tagger)

optimizer = nlp_blank.begin_training()
for _ in range(25):
    random.shuffle(train_data)
    losses = {}
    for text,annotations in train_data:
        nlp_blank.update([text], [annotations], sgd=optimizer, losses=losses)
    print(losses)

test_text = 'I like blue eggs'
doc = nlp_blank(test_text)
print('tags: ', [(x.text,x.tag_,x.pos_) for x in doc])
