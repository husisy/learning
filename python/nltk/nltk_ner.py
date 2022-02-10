import re

import nltk
from nltk.corpus import treebank, ieer, conll2002


z1 = treebank.tagged_sents() #token + POS
nltk.ne_chunk(z1[22]).draw()


_PATTERN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=_PATTERN):
        print(nltk.sem.rtuple(rel))


# Dutch
vnv = """
(
is/V|    # 3rd sing present and
was/V|   # past forms of the verb zijn ('be')
werd/V|  # and also present
wordt/V  # past of worden ('become)
)
.*       # followed by anything
van/Prep # followed by van ('of')
"""
VAN = re.compile(vnv, re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
    for r in nltk.sem.extract_rels('PER', 'ORG', doc, corpus='conll2002', pattern=VAN):
        print(nltk.sem.clause(r, relsym="VAN")) 