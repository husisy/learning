"""
reference: https://github.com/explosion/spacy/blob/master/examples/pipeline/multi_processing.py

Example of multi-processing with Joblib. Here, we're exporting
part-of-speech-tagged, true-cased, (very roughly) sentence-separated text, with
each "sentence" on a newline, and spaces between tokens. Data is loaded from
the IMDB movie reviews dataset and will be loaded automatically via Thinc's
built-in dataset loader.
"""
raise Exception('_pickle.PicklingError: Could not pickle the task to send it to the workers')

import os
import spacy
from pathlib import Path
from toolz import partition_all
from thinc.extra.datasets import imdb
from joblib import Parallel, delayed

from utils import next_tbd_dir

def is_sent_begin(word):
    return word.i==0 or (word.i>=2 and word.nbor(-1).text in ('.', '!', '?', '...'))

def represent_word(word):
    text = word.text
    if text.istitle() and is_sent_begin(word) and word.prob<word.doc.vocab[text.lower()].prob:
        text = text.lower()
    return text + '|' + word.tag_

def transform_texts(nlp, batch_id, texts, logdir):
    out_path = Path(logdir)/('{}.txt'.format(batch_id))
    print('processing batch {}'.format(batch_id))
    with out_path.open('w', encoding='utf8') as fid:
        for doc in nlp.pipe(texts):
            fid.write(' '.join(represent_word(w) for w in doc if not w.is_space))
            fid.write('\n')
    print('saved {} texts to {}.txt'.format(len(texts), batch_id))


logdir = next_tbd_dir()
hf_file = lambda *x,dir0=logdir: os.path.join(dir0, *x)

n_jobs = 4
batch_size = 1000
limit = 10000

nlp_sm = spacy.load('en_core_web_sm')
data_train,data_test = imdb()
texts,label = zip(*data_train[-limit:])
partitions = partition_all(batch_size, texts)
do = delayed(transform_texts)
executor = Parallel(n_jobs=n_jobs)
executor(do(nlp_sm, i, batch, logdir) for i,batch in enumerate(partitions))
