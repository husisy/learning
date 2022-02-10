# TF Performance

**WARNING @deprecated**: this is note from tf-1.0

[performance guide](https://www.tensorflow.org/performance/performance_guide)

## general best practices

1. `cuDNN` prefers `NCHW`, `TF` compiled with `Intel MKL` prefers `NCHW`
2. fused operation
   * `tf.image.decode_and_crop_jpeg`
   * `tf.layers.batch_normalization`
3. RNN
   * `tf.nn.dynmic_rnn`, `tf.nn.static_rnn`: shouldn't be performance difference at runtime; For the later, large unroll amounts can increase the graph size (long compile time)
   * `tf.contrib.cudnn_rnn`` is prefered (**an order faster and less memory**) but not support layer normalization
   * on CPUs / mobile devices (`tf.contrib.cudnn_rnn` not available), the fastest and memory efficient option is `tf.contrib.rnn.LSTMBlockFusedCell`
4. **build and install from source**

## optimizing for CPU

1. build and install from source
2. [Intel MKL-DNN](https://www.tensorflow.org/performance/performance_guide#tensorflow_with_intel_mkl_dnn)
3. [TensorFlow with Intel MKL-DNN](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)
4. Linux only
5. not work when using `--config=cuda`

## input pipeline

1. check
   * if GPU utilization is approaching `80-100%`
   * generate a timeline and look for large blocks of white space (waiting), see [XLA JIT tutorial](https://www.tensorflow.org/performance/xla/jit)
   * lack the CPU cycles to process the pipeline
   * spinning disks (150MB/sec), SATA SSDs (500MB/sec) PCIe SSDs(2000+MB/sec)
2. place the data input pipeline on CPU; default for `tf.estimator.Estimator` [ResNet example](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator/cifar10_main.py)
3. data source
   * local: DNN, SSD
   * remote: GCS, HDFS
4. preprocess large number of small files to `tfrecords`; For small `tfrecords` (<1GB), load the entire dataset into memory
5. use `tf.data` API, not use `feed_dict` or `queue_runner`
   * `.prefetch()`
   * `num_parallel_calls`
   * `tf.contrib.data.map_and_batch()`
   * `tf.contrib.data.parallel_interleave()`
   * `.cache()` after `.map()` if `map_fn` is time expensive
6. order of operation
   * `.map()`, `.batch()`
   * `.interleave() / .prefetch() / .shuffle()` maintain internal buffer of elements, if `map_fn()` generate elements of large size, recommand put `.map()` at the end of these kind op when memory is limited
   * `.shuffle()`, `.repeat()`, `tf.contrib.data.shuffle_and_repeat()`

```Python
def parse_fn(example):
    example_fmt = {
        "image": tf.FixedLengthFeature((), tf.string, ""),
        "label": tf.FixedLengthFeature((), tf.int64, -1)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.image.decode_image(parsed["image"])
    image = _augment_helper(image)
    return image, parsed["label"]

def input_fn():
    ds = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord") \
            .interleave(tf.data.TFRecordDataset) \
            .repeat() \
            .shuffle(1000) \
            .map(parse_fn, num_parallel_calls=2) \
            .batch(batch_size=FLAGS.batch_size) \
            .prefetch(n) #a single training step comsume n elements
    return ds
```

## Fixed Point Quantization

ops
