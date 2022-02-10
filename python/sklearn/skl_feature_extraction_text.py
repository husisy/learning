import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import next_tbd_dir

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-3: round(hfe(x,y,eps),5)

def skl_tfidf(z1=None):
    '''Term Frequency Inverse Document Frequency'''
    if z1 is None:
        z1 = ['hello world aha', 'hello world hello hello', 'aha aha aha']
    z2 = [x.split() for x in z1]

    v = TfidfVectorizer(strip_accents='unicode')
    z3 = v.fit_transform(z1)

    def hf1(sent, word_to_id=v.vocabulary_):
        ret = [0]*len(word_to_id)
        for x,y in Counter(sent).most_common():
            ret[word_to_id[x]] = y
        return ret
    term_freq = np.array([hf1(x) for x in z2])
    num_sample = len(z1) + 1 #smooth_idf
    doc_freq = np.sum(term_freq>0.5, axis=0) + 1 #smooth_idf
    tmp1 = term_freq * (np.log(num_sample/doc_freq) + 1)
    tf_idf = tmp1/np.sqrt(np.sum(tmp1**2, axis=1, keepdims=True))

    print('skl_tfidf:: sklearn vs np: ', hfe_r5(z3, tf_idf))

if __name__=='__main__':
    skl_tfidf()
