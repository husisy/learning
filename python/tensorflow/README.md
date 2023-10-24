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

```bash
# x86-64 cpu environment for tensorflow
micromamba create -n env-tf
micromamba install -y -n env-tf python=3.11 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs pytest-xdist
micromamba activate env-tf
pip install tensorflow

# x86-64 cuda118 tensorflow
micromamba create -y -n cuda118-tf
micromamba install -y -n cuda118-tf cudatoolkit=11.8 cudnn python=3.11 cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum cvxpy scs
micromamba activate cuda118-tf
pip install tensorflow
## replace <USERNAME> with your name
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/<USERNAME>/micromamba/envs/cuda118-tf/lib"
```

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
