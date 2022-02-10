# tensorflow

1. link
   * [documentation](https://www.tensorflow.org/alpha), `20190521`
   * [Get Started](https://www.tensorflow.org/alpha/tutorials/quickstart/beginner)
   * [Effective tf2](https://www.tensorflow.org/alpha/guide/effective_tf2)
   * [Request for Comments (RFC)](https://github.com/tensorflow/community)
   * [TensorFlow playground](http://playground.tensorflow.org)
2. 安装
   * `pip install tensorflow`
   * `pip install tensorflow-datasets`
   * `pip install tensorflow-hub`
3. 使用google colab
4. [Keras - overview](https://www.tensorflow.org/alpha/guide/keras/overview)
   * sequential model: `tf.keras.Sequential()`
   * functional API: `tf.keras.Model()` + `tf.keras.layers.Dense()`
   * model subclassing: `class MyModel(tf.keras.Model)` + `tf.keras.layers.Dense()`
5. [keras - custom layer](https://www.tensorflow.org/alpha/guide/keras/overview#custom_layers)
   * `__init__()`
   * `build()`
   * `call()`
   * `get_config()` and `from_config()`
6. Image Augmentation
   * [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
   * [Efficient DataFlow](http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html)
7. 偏见
   * 在`tf.keras.Model`中**禁止**使用`__call__`，**只**能使用`call`，`__call__`不能完全替代`call`的作用

TODO list

1. link
   * 优先阅读[documentation / guide/ customization](https://www.tensorflow.org/guide/eager)
   * [model zoo](https://github.com/tensorflow/models/tree/master/official/resnet)
   * [tensorflow example](https://github.com/tensorflow/examples)
   * [tensorflow dataset](https://github.com/tensorflow/datasets)
   * [tensorboard-documentation](https://www.tensorflow.org/tensorboard/get_started)
2. `tf_feature_column.py`
3. `tf.keras.optimizer.apply_gradient()`
4. `example/`
5. distributed training
6. tensorboard export graph, see `test_tf_keras_layers.test_tf_keras_layers_SimpleRNNCell()`
