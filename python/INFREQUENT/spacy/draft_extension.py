import os
import spacy
from spacy import displacy
from spacy.lang.en import English
from spacy.tokens import Doc, Span

from utils import next_tbd_dir


def overlap_extension():
    def overlap(doc1, doc2):
        tmp1 = set(x.text for x in doc1)
        return [x for x in doc2 if x.text in tmp1]
    if not Doc.has_extension('overlap'):
        Doc.set_extension('overlap', method=overlap)

    nlp_en = English()
    doc1 = nlp_en('Peach emoji is where it has always been.')
    doc1.ents = [Span(doc1, 0, 2, label=nlp_en.vocab.strings['PERSON'])]
    doc2 = nlp_en('Peach is the superior emoji.')
    print('overlap_extension:: doc1:    ', [x.text for x in doc1])
    print('overlap_extension:: doc2:    ', [x.text for x in doc2])
    print('overlap_extension:: overlap: ', doc1._.overlap(doc2))


def to_html_entension():
    def to_html(doc, style='dep'):
        return displacy.render(doc, style=style, page=True)
    if not Doc.has_extension('to_html'):
        Doc.set_extension('to_html', method=to_html)

    logdir = next_tbd_dir()
    hf_file = lambda *x: os.path.join(logdir, *x)
    nlp_en = English()
    doc1 = nlp_en('Peach emoji is where it has always been.')
    doc1.ents = [Span(doc1, 0, 2, label=nlp_en.vocab.strings['PERSON'])]
    html = doc1._.to_html(style='ent')
    path = hf_file('ent00.html')
    print('html file save to {}'.format(path))
    with open(path, 'w', encoding='utf-8') as fid:
        fid.write(html)


if __name__=='__main__':
    overlap_extension()
    to_html_entension()
