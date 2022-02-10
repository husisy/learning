import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def demo_tfrecords_basic():
    N0 = 100
    logdir = next_tbd_dir()
    hf_file = lambda *x: os.path.join(logdir, *x)
    X1 = np.random.randint(0, 3, size=(N0,))
    X2 = np.random.randint(0, 3, size=(N0,2))
    X3 = np.random.rand(N0).astype(np.float32)
    X4 = np.random.rand(N0,2).astype(np.float32)
    X5 = [str(x) for x in range(N0)]
    X6 = [(str(x),str(x+1)) for x in range(N0)]
    tfrecords_file = hf_file('test01.tfrecords')

    # write tfrecords
    with tf.io.TFRecordWriter(tfrecords_file) as writer:
        for ind1 in range(N0):
            example = tf.train.Example(features=tf.train.Features(feature={
                'X1': _int64_feature(X1[ind1]),
                'X2': _int64_list_feature(X2[ind1]),
                'X3': _float_feature(X3[ind1]),
                'X4': _float_list_feature(X4[ind1]),
                'X5': _bytes_feature(X5[ind1].encode()),
                'X6':_bytes_list_feature([x.encode() for x in X6[ind1]]),
            }))
            writer.write(example.SerializeToString())

    # read tfrecords
    def ds_decode_tfrecords(example_proto):
        example_fmt = {
            'X1': tf.io.FixedLenFeature([], tf.int64),
            'X2': tf.io.FixedLenFeature([2], tf.int64),
            'X3': tf.io.FixedLenFeature([], tf.float32),
            'X4': tf.io.FixedLenFeature([2], tf.float32),
            'X5': tf.io.FixedLenFeature([], tf.string),
            'X6': tf.io.FixedLenFeature([2], tf.string)
        }
        ret = tf.io.parse_single_example(example_proto, features=example_fmt)
        return ret['X1'],ret['X2'],ret['X3'],ret['X4'],ret['X5'],ret['X6']

    ds1 = tf.data.TFRecordDataset(tfrecords_file).map(ds_decode_tfrecords)
    tmp0 = list(zip(*list(iter(ds1))))
    X1_,X2_,X3_,X4_ = [np.stack([y.numpy() for y in x]) for x in tmp0[:4]]
    X5_ = [x.numpy().decode('utf-8') for x in tmp0[4]]
    X6_ = [[y.decode('utf-8') for y in x.numpy()] for x in tmp0[5]]

    assert np.all(X1==X1_) #int64
    assert np.all(X2==X2_) #int64
    assert hfe(X3,X3_) < 1e-4 #float32
    assert hfe(X4,X4_) < 1e-4 #float32
    assert all(y0==y1 for y0,y1 in zip(X5,X5_))
    assert all(z0==z1 for y0,y1 in zip(X5,X5_) for z0,z1 in zip(y0,y1))


def demo_tfrecords_varlength():
    N0 = 100
    min_len = 3
    max_len = 7
    logdir = next_tbd_dir()
    hf_file = lambda *x: os.path.join(logdir, *x)
    hf_str = lambda :''.join([chr(x) for x in np.random.randint(97,123,size=[np.random.randint(3,10)])])
    X1_len = np.random.randint(min_len,max_len,size=(N0,))
    X1 = [np.random.randint(0,100,size=[x]) for x in X1_len]
    X2_len = np.random.randint(min_len,max_len,size=(N0,))
    X2 = [np.random.rand(x) for x in X2_len]
    X3_len = np.random.randint(min_len,max_len,size=(N0,))
    X3 = [[hf_str() for _ in range(x)] for x in X3_len]
    tfrecords_file = hf_file('test01.tfrecords')

    # write
    with tf.io.TFRecordWriter(tfrecords_file) as writer:
        for ind1 in range(N0):
            example = tf.train.Example(features=tf.train.Features(feature={
                'X1': _int64_list_feature(X1[ind1]),
                'X2': _float_list_feature(X2[ind1]),
                'X3': _bytes_list_feature([x.encode() for x in X3[ind1]]),
            }))
            writer.write(example.SerializeToString())

    # read
    def tf_decode_tfrecords(example_proto):
        example_fmt = {
            'X1': tf.io.VarLenFeature(tf.int64),
            'X2': tf.io.VarLenFeature(tf.float32),
            'X3': tf.io.VarLenFeature(tf.string),
        }
        ret = tf.io.parse_single_example(example_proto, features=example_fmt)
        X1 = tf.sparse.to_dense(ret['X1']) #int64
        X2 = tf.sparse.to_dense(ret['X2']) #float32
        X3 = tf.sparse.to_dense(ret['X3']) #string
        return X1, X2, X3

    ds1 = tf.data.TFRecordDataset(tfrecords_file).map(tf_decode_tfrecords)
    tmp0 = list(zip(*list(iter(ds1))))
    X1_ = [x.numpy() for x in tmp0[0]]
    X2_ = [x.numpy() for x in tmp0[1]]
    X3_ = [[y.decode('utf-8') for y in x.numpy()] for x in tmp0[2]]
    assert all(np.all(x==y) for x,y in zip(X1,X1_))
    assert all(hfe(x,y)<1e-4 for x,y in zip(X2,X2_))
    assert all(z0==z1 for y0,y1 in zip(X3,X3_) for z0,z1 in zip(y0,y1))


if __name__=='__main__':
    demo_tfrecords_basic()
    demo_tfrecords_varlength()
