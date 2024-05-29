import numpy as np
import torch
import torch.distributed.pipeline.sync

class DummyModel00(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 5)

    def forward(self, x):
        x = self.fc(x)
        return x

model_fp32 = DummyModel00()
# create a quantized model instance
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
input_fp32 = torch.randn(3, 4)
res = model_int8(input_fp32)


fc1 = torch.nn.Linear(7, 5).cuda(0)
fc2 = torch.nn.Linear(5, 3).cuda(1)
model = torch.nn.Sequential(fc1, fc2)
model = torch.distributed.pipeline.sync.Pipe(model, chunks=8)
torch0 = torch.rand(16, 7).cuda(0)
output_rref = model(torch0) #ERROR Current RPC agent is not set!


torch0 = torch.rand(3, requires_grad=True)
torch1 = torch.rand(3, requires_grad=True) + 2
x0 = torch.distributions.Normal(loc=torch0, scale=torch1)
x1 = x0.sample([23])
assert not x1.requires_grad
x2 = x0.log_prob(x1)
assert x2.requires_grad
x3 = x0.rsample([23])
assert x0.has_rsample
assert x3.requires_grad
