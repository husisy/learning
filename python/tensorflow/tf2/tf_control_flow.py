import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

assert tf.executing_eagerly()

def tf_basic_if_else():
    print('# tf_basic_if_else')
    print('tf.constant(1)==1: ', tf.constant(1)==1)
    print('tf.constant(1)!=1: ', tf.constant(1)!=1)
    print('tf.equal(tf.constant(1), 1): ', tf.equal(tf.constant(1), 1))
    print('tf.constant(1)!=0: ', tf.constant(1)!=0)

    if tf.constant(1, dtype=tf.int64):
        print('bool(tf.constant(1)) is True')
    if not tf.constant(0, dtype=tf.int64):
        print('bool(tf.constant(0)) is False')


def np_collatz_conjecture(x):
    assert isinstance(x, int) and x>1
    ret = 0
    while x!=1:
        x = (x//2) if (x%2==0) else (3*x+1)
        ret += 1
    return ret

def tf_collatz_conjecture(x):
    assert isinstance(x, int) and x>1
    x = tf.convert_to_tensor(x, dtype=tf.int64)
    ret = 0
    while int(x)!=1:
        x = (x//2) if (int(x)%2==0) else (3*x+1)
        ret += 1
    return ret

def tf_while_if_else_collatz():
    # Collatz conjecture: https://en.wikipedia.org/wiki/Collatz_conjecture
    tmp1 = all(np_collatz_conjecture(x)==int(tf_collatz_conjecture(x)) for x in range(2, 30))
    print('tf_while_if_else_collatz: ', tmp1)


if __name__ == "__main__":
    tf_basic_if_else()
    print()
    tf_while_if_else_collatz()
