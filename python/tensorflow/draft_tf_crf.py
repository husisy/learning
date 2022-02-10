import numpy as np
import tensorflow as tf

hfe = lambda x,y,eps=1e-3:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)))

# WARNING tf-1.0-contrib.crf have been migrated to tfa.text, see https://github.com/tensorflow/addons
# TODO migrate the code below was written to tf-2.0

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


def _np_viterbi_proba(N1=5, N2=7):
    state_lg = np.random.rand(N1,N2)
    trans_proba_lg = np.random.rand(N2,N2)
    z1 = np_viterbi(state_lg, trans_proba_lg)
    return np.exp(z1[1]-z1[2])


def np_CRFForward(state_lg, path, sequence_length, trans_proba_lg):
    '''
    xxx_lg = np.log(xxx)

    state_lg(np,float,(N0,N1,N2))
    path(np,int,(N0,N1,))
    sequence_length(np,int,(N0,))
    trans_proba_lg(np,float,(N2,N2))
    (ret1)unary_score_lg(float)
    (ret2)binary_score_lg(float)
    (ret3)norm_lg(float)
    '''
    N0,N1,N2 = state_lg.shape

    tmp1 = np.reshape(np.arange(N0*N1)*N2 + path.reshape(-1), (N0,N1))
    tmp2 = state_lg.reshape(-1)[tmp1].reshape((N0,N1))
    tmp3 = np.stack([np.arange(N1)<x for x in sequence_length], axis=0)
    unary_score_lg = np.sum(tmp2*tmp3, axis=1)

    tmp1 = path[:,:-1]*N2 + path[:,1:]
    tmp2 = np.stack([np.arange(1,N1)<x for x in sequence_length], axis=0)
    binary_score_lg = np.sum(trans_proba_lg.reshape(-1)[tmp1] * tmp2, axis=1)

    norm_lg = np.array([np_viterbi(x[:y], trans_proba_lg)[2] for x,y in zip(state_lg, sequence_length)])
    return unary_score_lg, binary_score_lg, norm_lg


def tf_crf_score(N0=3, N1=5, N2=7):
    np1 = np.random.randn(N0, N1, N2).astype(np.float32) #state_lg
    np2 = np.random.randint(0, N2, size=(N0,N1))# path
    np3 = np.random.randint(max(N1-2,1), N1, size=(N0,)) #sequence_length
    np4 = np.random.randn(N2, N2).astype(np.float32) #trans_p

    np5,np6,np7 = np_CRFForward(np1,np2,np3,np4)
    np8 = np5 + np6
    np9 = np8 - np7

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf.constant(np3)
    tf4 = tf.constant(np4)
    tf5 = tf.contrib.crf.crf_unary_score(tf2, tf3, tf1)
    tf6 = tf.contrib.crf.crf_binary_score(tf2, tf3, tf4)
    tf7 = tf.contrib.crf.crf_log_norm(tf1, tf3, tf4)
    tf8 = tf.contrib.crf.crf_sequence_score(tf1, tf2, tf3, tf4)
    tf9,_ = tf.contrib.crf.crf_log_likelihood(tf1, tf2, tf3, tf4)
    with tf.Session() as sess:
        tf5_,tf6_,tf7_,tf8_,tf9_ = sess.run([tf5,tf6,tf7,tf8,tf9])
    print('tf_crf_score unary:: np vs tf: ', hfe(np5, tf5_))
    print('tf_crf_score binary:: np vs tf: ', hfe(np6, tf6_))
    print('tf_crf_score log_norm:: np vs tf: ', hfe(np7, tf7_))
    print('tf_crf_score sequence:: np vs tf: ', hfe(np8, tf8_))
    print('tf_crf_score log likelihood:: np vs tf: ', hfe(np9, tf9_))


def tf_crf_CrfForwardRnnCell(N0=3, N2=7):
    np1 = np.random.rand(N0, N2)
    np2 = np.random.rand(N0, N2)
    np3 = np.random.rand(N2, N2)
    tmp1 = np2[:,:,np.newaxis] + np3
    tmp2 = np.max(tmp1, axis=1)
    np4 = np1 + np.log(np.sum(np.exp(tmp1 - tmp2[:,np.newaxis]), axis=1)) + tmp2
    np5 = np.log(np.matmul(np.exp(np2), np.exp(np3))) + np1

    tf1 = tf.constant(np1)
    tf2 = tf.constant(np2)
    tf3 = tf.constant(np3)
    tf4,_ = tf.contrib.crf.CrfForwardRnnCell(tf3)(tf1, tf2)
    with tf.Session() as sess:
        tf4_ = sess.run(tf4)
    print('tf_crf_CrfForwardRnnCell_1:: np vs tf: ', hfe(np4, tf4_))
    print('tf_crf_CrfForwardRnnCell_2:: np vs tf: ', hfe(np5, tf4_))


def tf_crf_decode(N0=3, N1=5, N2=7):
    np1 = np.random.randn(N0, N1, N2).astype(np.float32) #state_lg
    # np2 = np.random.randint(0, N2, size=(N0,N1))# path
    np3 = np.random.randint(max(N1-2,1), N1, size=(N0,)) #sequence_length
    np4 = np.random.randn(N2, N2).astype(np.float32) #trans_p

    tmp1 = [np_viterbi(x[:y], np4) for x,y in zip(np1,np3)]
    np5 = np.stack([np.pad(x,[[0,N1-len(x)]], mode='constant') for x,_,_ in tmp1])
    np6 = np.array([x for _,x,_ in tmp1])

    tf1 = tf.constant(np1)
    # tf2 = tf.constant(np2)
    tf3 = tf.constant(np3, dtype=tf.int32)
    tf4 = tf.constant(np4)
    tf5,tf6 = tf.contrib.crf.crf_decode(tf1, tf4, tf3)
    with tf.Session() as sess:
        tf5_,tf6_ = sess.run([tf5,tf6])
    print('tf_crf_decode path:: np vs tf: ', hfe(np5,tf5_))
    print('tf_crf_decode score:: np vs tf: ', hfe(np6,tf6_))
