import os
import pickle
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import brown, nps_chat, conll2000, treebank
from nltk import word_tokenize, pos_tag
from nltk import DefaultTagger, RegexpTagger, UnigramTagger, BigramTagger
from nltk import NaiveBayesClassifier, DecisionTreeClassifier

from utils import next_tbd_dir

# nltk.help.upenn_tagset()
# nltk.help.upenn_tagset('RB')
# nltk.corpus.treebank.readme()


raw = 'And now for something completely different'
# raw = 'They refuse to permit us to obtain the refuse permit'
pos = pos_tag(word_tokenize(raw))


raw = '''The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./.'''
token = raw.split()
pos = [nltk.tag.str2tuple(x) for x in token]


pos = treebank.tagged_words()
tmp1 = Counter(nltk.bigrams(x for _,x in pos)).most_common()
[(x,z) for (x,y),z in tmp1 if y=='NN' or y=='NNS']
tmp1 = [x for (x,y) in pos if y.startswith('VB')]
Counter(tmp1).most_common()[:20]


# pos data
brown_sent = brown.sents(categories='news')
brown_pos = brown.tagged_sents(categories='news')
train_data,val_data = train_test_split(brown_pos, test_size=0.1)
logdir = next_tbd_dir()
hf_file = lambda *x,_dir=logdir: os.path.join(_dir, *x)


# DefaultTagger
default_tagger = DefaultTagger('NN')
default_tagger.tag(word_tokenize('I do not like green eggs and ham, I do not like them Sam I am!'))
default_tagger.tag(brown_sent[3])
default_tagger.evaluate(brown_pos)


# RegexpTagger
_POS_PATTERN = [
    ('.*ing$', 'VBG'), #gerunds
    ('.*ed$', 'VBD'), #simple past
    ('.*es$', 'VBZ'), #3rd singular present
    ('.*ould', 'MD'), #modals
    (r".*\'s$", 'NN$'), #possesssive nouns
    ('.*s$', 'NNS'), #plural nouns
    ('^-?[0-9]+(.[0-9]+)?$', 'CD'), #cardinal numbers
    ('.*', 'NN'), #nouns (default)
]
regexp_tagger = RegexpTagger(_POS_PATTERN)
regexp_tagger.tag(brown_sent[3])
regexp_tagger.evaluate(brown_pos)


# UnigramTagger
tmp1 = Counter(y for x in brown_sent for y in x).most_common(100)
tmp2 = set(x for x,_ in tmp1)
tmp3 = defaultdict(list)
for sent in brown_pos:
    for x,y in sent:
        if x in tmp2:
            tmp3[x].append(y)
tmp4 = {x:Counter(y).most_common(1)[0][0] for x,y in tmp3.items()}
likely_tagger = UnigramTagger(model=tmp4, backoff=DefaultTagger('NN'))
likely_tagger.tag(brown_sent[3])
likely_tagger.evaluate(brown_pos)
# UnigramTagger - analysis: see http://www.nltk.org/book/ch05.html  4.3 The Lookup Tagger


# UnigramTagger
unigram_tagger = UnigramTagger(train=train_data)
unigram_tagger.evaluate(val_data)


# BigramTagger + serialization
bigram_tagger = BigramTagger(train_data)
print(bigram_tagger.evaluate(val_data))
with open(hf_file('tagger.pkl'), 'wb') as fid:
    pickle.dump(bigram_tagger, fid)
with open(hf_file('tagger.pkl'), 'rb') as fid:
    z1 = pickle.load(fid)
print(bigram_tagger.evaluate(val_data))


# Default + Unigram + Bigram
t0 = DefaultTagger('NN')
t1 = UnigramTagger(train_data, backoff=t0)
t2 = BigramTagger(train_data, backoff=t1)
t2.evaluate(val_data)

label = [x for sent in brown_pos for _,x in sent]
predict = [x for sent in brown_sent for _,x in t2.tag(sent)]
# print(nltk.ConfusionMatrix(label, predict)) #what a mess


# DecisionTreeClassifier
common_suffix = [x for x,_ in Counter(y for x in brown.words() for y in [x[-1:],x[-2:],x[-3:]]).most_common(20)]
data = brown.tagged_words(categories='news')
train_data,val_data = train_test_split(data, test_size=0.1)
hf1 = lambda token,common_suffix=common_suffix: {('endswith_'+x):token.endswith(x) for x in common_suffix}
clf = DecisionTreeClassifier.train([(hf1(x),y) for x,y in train_data])
print('acc: ', sum(clf.classify(hf1(x))==y for x,y in val_data)/len(val_data))
# clf.pseudocode()


# NaiveBayesClassifier
data = brown.tagged_sents(categories='news')
train_data,val_data = train_test_split(data, test_size=0.1)
def hf1(sentence, ind1):
    return {
        'suffix_1': sentence[ind1][-1:],
        'suffix_2': sentence[ind1][-2:],
        'suffix_3': sentence[ind1][-3:],
        'previous_word': '<START>' if ind1==0 else sentence[ind1-1][0],
    }
hf2 = lambda data: [(hf1(list(zip(*sent))[0],ind1),x) for sent in data for ind1,(_,x) in enumerate(sent)]
clf = NaiveBayesClassifier.train(hf2(train_data))
tmp1 = hf2(val_data)
print('acc: ', sum(clf.classify(x)==y for x,y in tmp1)/len(tmp1))


# performance limitation
z1 = defaultdict(list)
for sent in brown_pos:
    for x,y,z in nltk.trigrams(sent):
        z1[(x[1],y[1],z[0])].append(z[1])
tmp1 = sum(len(x) for x in z1.values() if len(set(x))>1)
tmp2 = sum(len(x) for x in z1.values())
print('ambiguous: {}'.format(tmp1/tmp2))


# transformation-based tagging
dir(nltk.tag.brill) #TODO


