import os
import string
import random
import numpy as np

import sklearn_crfsuite


def np_viterbi(state_lg, trans_proba_lg):
    '''
    xxx_lg = np.log(xxx)

    state_lg(np,float,(N1,N2))
    trans_proba_lg(np,float,(N2,N2))
    (ret1)path(list/int)
    (ret2)score_lg(float)
    (ret3)norm_lg(float)
    '''
    def np_logsumexp(x):
        tmp1 = np.max(x, axis=0)
        return np.log(np.sum(np.exp(x-tmp1[np.newaxis]), axis=0)) + tmp1
    max_lg = state_lg[0]
    sum_lg = state_lg[0]
    hist_ind = []
    for state_i_lg in state_lg[1:]:
        tmp1 = max_lg[:,np.newaxis] + trans_proba_lg + state_i_lg
        max_lg = np.max(tmp1, axis=0)
        hist_ind.append(np.argmax(tmp1, axis=0))
        sum_lg = np_logsumexp(sum_lg[:,np.newaxis]+trans_proba_lg) + state_i_lg
    path = [max_lg.argmax()]
    for x in hist_ind[::-1]: path.append(x[path[-1]])
    path = path[::-1]
    score_lg = max_lg.max()
    norm_lg = np_logsumexp(sum_lg)
    return path, score_lg, norm_lg


def generate_integer_dataset():
    bag_of_word = ['word'+str(ind1) for ind1 in range(10)]
    token_property = ['prop'+str(ind1) for ind1 in range(5)]
    ner = ['ner'+str(ind1) for ind1 in range(5)]
    word_property = [(x,random.choice(token_property)) for x in bag_of_word]

    def random_sentence(min_=5, max_=10, word_property=word_property, ner=ner):
        num1 = random.randint(min_, max_)
        return [(*random.choice(word_property), random.choice(ner)) for _ in range(num1)]

    sentence = [random_sentence() for _ in range(20)]
    return bag_of_word,token_property,ner,word_property,sentence


def sent2features(sent):
    def word2features(sent, ind1):
        ret = {}
        ret['bias'] = 1
        ret['word'] = sent[ind1][0]
        ret['prop'] = sent[ind1][1]
        if ind1 > 0:
            ret['-1:word'] = sent[ind1-1][0]
            ret['-1:prop'] = sent[ind1-1][1]
        else:
            ret['BOS'] = True
        if ind1 < len(sent)-1:
            ret['+1:word'] = sent[ind1+1][0]
            ret['+1:prop'] = sent[ind1+1][1]
        else:
            ret['EOS'] = True
        return ret
    return [word2features(sent, ind1) for ind1 in range(len(sent))]


def skl_crf_predict_path():
    bag_of_word,token_property,ner,word_property,sentence = generate_integer_dataset()
    tmp1 = [
        {'BOS','EOS','bias'},
        {'word:'+str(x) for x in bag_of_word},
        {'prop:'+str(x) for x in token_property},
        {'-1:word:'+str(x) for x in bag_of_word},
        {'-1:prop:'+str(x) for x in token_property},
        {'+1:word:'+str(x) for x in bag_of_word},
        {'+1:prop:'+str(x) for x in token_property},
    ]
    attribute = {y for x in tmp1 for y in x}

    X_train = [sent2features(x) for x in sentence]
    y_train = [[y[2] for y in x] for x in sentence]
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)

    trans_proba_lg = np.array([[crf.transition_features_[x,y] for y in ner] for x in ner])
    attribute_to_weights = {(x,y):crf.state_features_.get((x,y),0) for x in attribute for y in ner}

    hf0 = lambda k,v: k if k in {'EOS','bias','BOS'} else k+':'+v
    hf1 = lambda w: [sum(attribute_to_weights[(hf0(k,v), x)] for k,v in w.items()) for x in ner]
    state_proba_lg = [np.array([hf1(w) for w in s]) for s in X_train]
    crf_path_ = [np_viterbi(x, trans_proba_lg)[0] for x in state_proba_lg]

    tmp1 = {y:x for x,y in enumerate(ner)}
    crf_path = [[tmp1[y] for y in x] for x in crf.predict(X_train)]

    print('crf_predict_path:: np vs sklcrf: ', all(x==y for x,y in zip(crf_path,crf_path_)))
