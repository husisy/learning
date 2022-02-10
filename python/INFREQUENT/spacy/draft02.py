import spacy
import numpy as np

nlp_sm = spacy.load('en_core_web_sm')
nlp_md = spacy.load('en_core_web_md')


#
text = 'dog cat banana afskfsd'
doc = nlp_md(text) #doc = nlp_sm(text)
str_fmt = '{:>15} {:>15} {:>15} {:>15} {:>15}'
print(str_fmt.format('similarity',*[x.text for x in doc]))
for x in doc:
    print(str_fmt.format(x.text, *[str(round(x.similarity(y), 3)) for y in doc]))

print('')
str_fmt = '{:>15} {:>15} {:>15} {:>15}'
print(str_fmt.format('text', 'has_vector', 'vector_nrom', 'is_oov')) #out of vocabulary
for x in doc:
    print(str_fmt.format(x.text, x.has_vector, round(float(x.vector_norm),3), x.is_oov))


# numpy vector
doc = nlp_md('apple')
print(type(doc[0].vector), doc[0].vector.shape)
print(type(nlp_md.vocab.vectors.data), nlp_md.vocab.vectors.data.shape)
ind1 = nlp_md.vocab.vectors.key2row[nlp_md.vocab.strings['apple']]
assert np.shares_memory(nlp_md.vocab.vectors.data[ind1], doc[0].vector)


# prune word vector
if input('type YES to confirm prune_vectors (SLOW): ')=='YES':
    n_vector_to_keep = 50000
    removed_words = nlp_md.vocab.prune_vectors(n_vector_to_keep)
    assert len(nlp_md.vocab.vectors) <= n_vector_to_keep  # unique vectors have been pruned
    assert nlp_md.vocab.vectors.n_keys > n_vector_to_keep  # but not the total entries
