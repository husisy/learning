# pytorch

1. link
   * [offiical site](http://pytorch.org/)
   * [deep learning with pytorch: a 60 minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
   * [tutorials](http://pytorch.org/tutorials/)
   * [documentation - multi gpu, data parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
   * [github/conv-arithmetic](https://github.com/vdumoulin/conv_arithmetic)
2. install
   * `conda install -c pytorch pytorch torchvision cudatoolkit=10.1`
   * `pip install torch torchvision`
3. 建议阅读
   * [official-site/tutorial/learning pytorch](https://pytorch.org/tutorials/#learning-pytorch)下的四个模块
   * [official-site/docs/Notes](https://pytorch.org/docs/stable/index.html)
4. `channel_last` is able to achieve over `22%` performance gains with apex, see [github/apex](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#performance-gains)
5. 偏见
   * **禁止**使用`torch0.data`属性，使用`torch0.detach()`替代，该属性只用作向后兼容， [link](https://pytorch.org/docs/stable/onnx.html#avoid-using-data-field)
6. extension
   * [github/torchquad](https://github.com/esa/torchquad) integral of function

TODO

1. test data loader performance
2. [github/pytorch/examples](https://github.com/pytorch/examples)

custom optimizer

1. link
   * [pytorch discuss / regarding implementation of optimization algorithm](https://discuss.pytorch.org/t/regarding-implementation-of-optimization-algorithm/20920/2)
   * [pytorch doc / SGD source code](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD)
   * [pytorch doc / Adam source code](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam)
   * [mcneela / implementating a novel optimizer from scratch](http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html)
2. see `draft_custom_optimizer.py`

## GAN

1. link
   * [pytorch tutorial / DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
   * [github /ganhacks](https://github.com/soumith/ganhacks)
   * [github / gan-awesome-applications](https://github.com/nashory/gans-awesome-applications)
   * [GAN lab](https://poloclub.github.io/ganlab/)
   * [GAN Dissection](https://gandissect.csail.mit.edu/)
   * [pytorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

## Adversarial Example Generation

1. link
   * [pytorch tutorial / adversarial example generation](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

## jit

1. link
   * [torch-doc-link](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) Introduction to TorchScript
   * [torch-doc-link](https://pytorch.org/tutorials/advanced/cpp_export.html) Loading a TorchScript Model in cpp
2. concept
   * model authoring
