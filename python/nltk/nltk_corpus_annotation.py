from collections import Counter, defaultdict

import nltk
from nltk.tree import Tree
from nltk.corpus import treebank, treebank_chunk, brown

# tagset
#  brown: POS
#  PennTreeBank(PTB): POS, syntactic_bracketing, chunking(NP)


# treebank: PTB_POS + PTB_syntactic_bracketing
file_list = list(treebank.fileids())
sent = treebank.parsed_sents(file_list[0])
set(y.label() for x in sent for y in x.subtrees(lambda _x: isinstance(_x, Tree))) - {'S'}
sent[0].leaves()
sent[0].pos()


# treebank_chunk: PTB_POS + PTB_chunk
chunk_sent = treebank_chunk.chunked_sents()
set(y.label() for x in chunk_sent for y in x.subtrees(lambda _x: isinstance(_x, Tree))) - {'S'}


# brown
token = brown.words(categories='learned')
Counter(y for (x,y) in nltk.bigrams(token) if x=='often').most_common()

pos = brown.tagged_words(categories='learned')
Counter(y[1] for (x,y) in nltk.bigrams(pos) if x[0]=='often').most_common()

sent = brown.tagged_sents()
hf1 = lambda x,y,z: x[1].startswith('V') and y[1]=='TO' and z[1].startswith('V')
Counter(y for x in sent for y in nltk.trigrams(x) if hf1(*y)).most_common(20)

pos = brown.tagged_words(categories='news', tagset='universal')
z1 = defaultdict(list)
for x,y in pos: z1[x.lower()].append(y)
for key,value in z1.items():
    tmp1 = Counter(value)
    if len(tmp1)>3:
        print(key, ': ', ' '.join(x[0] for x in tmp1.most_common()))


