import os
import math
import random
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from utils import next_tbd_dir

def tf_wordvec_visualization(logdir, id_to_wordvec, id_to_word, **kwargs):
    '''
    logdir(str): path, must be not exist or empty
    id_to_wordvec(dict)
        id(int) -> wordvec(np,float,(N1,))
    id_to_word(dict)
        id(int) -> word(str)
    kwargs
        id_to_info(dict)
            id(int) -> ?(str)
    '''
    assert (not os.path.exists(logdir)) or len(os.listdir(logdir))==0, 'logdir must be not created before nor empty'
    assert all(x.startswith('id_to_') for x in kwargs.keys()), 'all kwargs keys should startswith id_to_'
    if not os.path.exists(logdir): os.makedirs(logdir)
    hf_file = lambda *x,dir0=logdir: os.path.join(dir0, *x)
    kwargs = [(k,v) for k,v in kwargs.items()]

    tmp1 = [set(x.keys()) for _,x in kwargs]
    id_set = list(set.intersection(set(id_to_wordvec.keys()), set(id_to_word.keys()), *tmp1))
    wordvec = np.stack([id_to_wordvec[x] for x in id_set], axis=0)
    with open(hf_file('metadata.tsv'), 'wb') as fid:
        def hfid_write(*x):
            x = ['<Space>' if y.isspace() else y for y in x]
            fid.write(('\t'.join(x) + '\n').encode('utf-8'))
        hfid_write('word', *[x[6:] for x,_ in kwargs])
        for id_ in id_set:
            hfid_write(id_to_word[id_], *[x[id_] for _,x in kwargs])

    with tf.Graph().as_default() as tfG:
        tf1 = tf.get_variable('embedding', dtype=tf.float32, shape=wordvec.shape)

    with tf.Session(graph=tfG) as sess:
        with tf.summary.FileWriter(logdir, tfG) as writer:
            sess.run(tf.assign(tf1, wordvec))
            config = projector.ProjectorConfig()
            embed = config.embeddings.add()
            embed.tensor_name = tf1.name
            embed.metadata_path = 'metadata.tsv'
            projector.visualize_embeddings(writer, config)
            tf.train.Saver().save(sess, hf_file('model.ckpt'))
    print('run "tensorboard --logdir {}" to see word embedding'.format(logdir))


if __name__=='__main__':
    logdir = next_tbd_dir()
    nlp_md = spacy.load('en_core_web_md')

    N0 = int(input('input number of words to visualize [default 10000]: ') or 10000)
    tmp1 = [x for x in nlp_md.vocab.strings if nlp_md.vocab.has_vector(x)]
    id_to_word = dict(enumerate(random.sample(tmp1, N0)))
    id_to_wordvec = {k:nlp_md.vocab.get_vector(v) for k,v in id_to_word.items()}
    hf1 = lambda v: str(round(math.exp(nlp_md.vocab[v].prob)*N0, 5))
    id_to_frequency = {k:hf1(v) for k,v in id_to_word.items()}

    tf_wordvec_visualization(logdir, id_to_wordvec, id_to_word, id_to_frequency=id_to_frequency)
