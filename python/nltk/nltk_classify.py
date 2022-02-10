import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.corpus import names, movie_reviews, brown, treebank_raw, nps_chat
from nltk import NaiveBayesClassifier, DecisionTreeClassifier

# sequence classification TODO
# Hidden Markov Model TODO
# Maximum Entripy Markov Model TODO
# Linear-Chain Conditional Random Field Models TODO


# NaiveBayesClassifier
data = [(x,'male') for x in names.words('male.txt')] + [(x,'female') for x in names.words('female.txt')]
train_data,val_data = train_test_split(data, test_size=0.3)
hf1 = lambda x: {'last_letter':x[-1]}
clf = NaiveBayesClassifier.train([(hf1(x),y) for x,y in train_data])
clf.classify(hf1('Neo')) #'Trinity'
predict = [clf.classify(hf1(x)) for x,_ in val_data]
relation = set((x[-1],y) for (x,_),y in zip(val_data, predict))
acc = sum(x==y for (_,x),y in zip(val_data,predict))/len(predict)
clf.show_most_informative_features()
# nltk.classify.apply_features


# IMDB movie review
data = [(list(movie_reviews.words(y)),x) for x in movie_reviews.categories() for y in movie_reviews.fileids(x)]
train_data,val_data = train_test_split(data, test_size=0.1)
all_word = random.sample(set(x.lower() for x in movie_reviews.words()), 2000)
def hf2(token, all_word=all_word):
    token = set(token)
    return {('contain_'+x):(x in token) for x in all_word}
clf = nltk.NaiveBayesClassifier.train([(hf2(x),y) for x,y in train_data])
print('acc: ', sum(clf.classify(hf2(x))==y for x,y in val_data)/len(val_data))
clf.show_most_informative_features()


# identify dialogue act type
data = [(x.text,x.get('class')) for x,_ in zip(nps_chat.xml_posts(), range(10000))]
train_data,val_data = train_test_split(data, test_size=0.1)
hf1 = lambda post: {('word_'+x.lower()):True for x in word_tokenize(post)}
clf = NaiveBayesClassifier.train([(hf1(x),y) for x,y in train_data])
print('acc: ', sum(clf.classify(hf1(x))==y for x,y in val_data)/len(val_data))


# Recognize Textual Entailment (RTE) TODO

