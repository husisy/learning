import spacy

nlp_sm = spacy.load('en_core_web_sm')


# multi-threaded generator
texts = ['One documents.', '...', 'Lots of documents']
gen = (texts[ind1 % len(texts)] for ind1 in range(1000))
for ind1,doc in enumerate(nlp_sm.pipe(gen, batch_size=50, n_threads=4)):
    assert doc.is_parsed
    if ind1==100: break
