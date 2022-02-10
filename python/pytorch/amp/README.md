# mixed precision

1. link
   * [documentation / automatic mixed precision examples](https://pytorch.org/docs/master/notes/amp_examples.html)
   * [github / mixed-precision-pytorch](https://github.com/suvojit-0x55aa/mixed-precision-pytorch)
   * [arxiv / mixed precision training](https://arxiv.org/abs/1710.03740)
   * [nvidia-GTC/training-neural-networks-with-mixed-precision-real-example](http://on-demand.gputechconf.com/gtc/2018/video/S81012/)
2. 偏见
   * `torch.cuda.amp.autocast()`应该包括前向过程（包括计算loss），**不应该**包括反向过程
   * `float64`以及non-floating-point类型的算子不支持autocast
   * 仅CUDA算子支持autocast
   * 用`F.binary_cross_entropy_with_logits`，**禁止**使用`F.binary_cross_entropy`，**禁止**使用`torch.nn.BCELoss()`
3. Only out-of-place ops and tensor methods are eligible. In-place variants and calls that explicitly supply an `out=...` tensors are allowed in autocast-enabled regions, but won't go through autocasting
   * `a.addmm(b,c)` 可以
   * `a.addmm_(b,c)`, `a.addmm(b, c, out=d)`不可
4. Op-specific behavior
   * ops, see [link](https://pytorch.org/docs/stable/amp.html#autocast-op-reference)
5. underflow issue, gradient scaler
6. the optimal scale factor is the largest factor that can be used without incurring inf or NaN gradient values
