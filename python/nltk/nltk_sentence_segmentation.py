from sklearn.model_selection import train_test_split

from nltk import sent_tokenize
from nltk.corpus import gutenberg, treebank_raw
from nltk import NaiveBayesClassifier, DecisionTreeClassifier

raw = gutenberg.raw('chesterton-thursday.txt')
sent = sent_tokenize(raw)


# sentence segmentation
def hf1(token, ind1):
    return {
        'next_word_cap': token[ind1+1].isupper(),
        'prev_word': token[ind1-1].lower(),
        'prev_word_single': len(token[ind1-1])==1,
        'curr': token[ind1],
    }
def hf2(data):
    token,label = zip(*data)
    return [(hf1(token,ind1),label[ind1]) for ind1 in range(1,len(token)-1) if token[ind1] in '.?!']
tmp1 = [x for sent in treebank_raw.sents() for x in zip(sent, [0]*(len(sent)-1)+[1])]
data = hf2(tmp1)
train_data,val_data = train_test_split(data, )
clf = NaiveBayesClassifier.train(train_data)
print('acc: ', sum(clf.classify(x)==y for x,y in val_data)/len(val_data))
