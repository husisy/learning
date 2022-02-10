import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

@tf.function
def hf1(x, y):
    return tf.nn.relu(tf.matmul(x, y))

logdir = next_tbd_dir()
writer = tf.summary.create_file_writer(logdir)

tf1 = tf.random.normal((3, 4))
tf2 = tf.random.normal((4, 5))

tf.summary.trace_on(graph=True, profiler=True)
tf3 = hf1(tf1, tf2)
with writer.as_default():
    tf.summary.trace_export(name='hf1_trace', step=0, profiler_outdir=logdir)

