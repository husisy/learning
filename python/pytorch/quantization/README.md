# pytorch quantization

1. link
   * [pytorch-blog/introduction-to-quantization](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
   * [documentation/quantization](https://pytorch.org/docs/stable/quantization.html)
   * [documentation/FX-Graph-Mode-Quantization-User-Guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)
   * [documentation/dynamic-quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) [advanced](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html)
   * tutorial中的模型下载链接列表 [link](https://github.com/pytorch/tutorials/blob/master/Makefile)
2. 主要用于推理，Network Architecture Search (NAS)，仅支持forward pass
3. qint8
4. quantization aware training
5. Eager Mode Quantization (beta), FX Graph Mode Quantization (prototype)
6. `torch.fx`
7. symbolically traceable
8. type
   * dynamic quantization: weights quantized with activations read/stored in floating point and quantized for compute
   * static quantization: weights quantized, activations quantized, calibration required post training, also known as Post Training Quantization (PTQ)
   * static quantization aware training: weights quantized, activations quantized, quantization numerics modeled during training
