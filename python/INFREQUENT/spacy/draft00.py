import spacy
from spacy.lang.en import English
from spacy.pipeline import SentenceSegmenter

nlp_en = spacy.load('en')
nlp_sm = spacy.load('en_core_web_sm')
nlp_md = spacy.load('en_core_web_md')

text = [
    'This is a sentence. This is another sentence.',
    'this is a sentence...hello...and another sentence.',
    'This is a sentence\n\nThis is another sentence\nAnd more',
]

# built-in
doc = nlp_sm(text[0])
for sent in doc.sents:
    print(sent.text)


# customize
doc = nlp_sm(text[1])
print('before: ', [sent.text for sent in doc.sents])

def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text=='...':
            doc[token.i+1].is_sent_start = True
    return doc
nlp_sm.add_pipe(set_custom_boundaries, before='parser')
doc = nlp_sm(text[1])
print('after:  ', [sent.text for sent in doc.sents])

nlp_sm = spacy.load('en_core_web_sm')


# rule-based pipeline
nlp01 = English()
sbd = nlp01.create_pipe('sentencizer')
nlp01.add_pipe(sbd)
doc = nlp01(text[0])
for sent in doc.sents:
    print(sent.text)


# custom rule-based strategy
def split_on_newlines(doc):
    start = 0
    seen_newline = False
    for word in doc:
        if seen_newline and not word.is_space:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text=='\n':
            seen_newline = True
    if start < len(doc):
        yield doc[start:len(doc)]

nlp01 = English()
sbd = SentenceSegmenter(nlp01.vocab, strategy=split_on_newlines)
nlp01.add_pipe(sbd)
doc = nlp01(text[2])
for sent in doc.sents:
    print([token.text for token in sent])
