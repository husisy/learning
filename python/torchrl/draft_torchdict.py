import numpy as np
import torch
import tensordict

# shape
z0 = tensordict.TensorDict({'zeros': torch.zeros(2,3,4), 'ones': torch.ones(2,3,4,5)}, batch_size=[2,3,4])
z0.shape #2,3,4
z0.batch_size #same as z0.shape
z0.numel() #2*3*4
z0['zeros']
z0['rand'] = torch.rand(2, 3, 4)
z0[:,::2] #only for batch dimension
x0, x1, x2 = z0.unbind(1) #shape=2,4
# z0.to('cuda')
z1 = z0.reshape(-1) #shape=24
x0,x1 = z0.split([3, 1], dim=2) #shape=(2,3,3) shape=(2,3,1)
z2 = torch.cat([z0, z0.clone()], dim=0) #shape=4,3,4
z3 = z0.expand(2, *z0.batch_size) #shape=(2,2,3,4) tile


# key value
z0 = tensordict.TensorDict({'a': torch.rand(2,3), 'b': torch.rand(2,4)}, batch_size=2)
z0['c'] = torch.rand(2,5)
z0.del_('c') #in-place
list(z0.keys()) #z0.keys() is a tensordict datatype (bad-design)
list(z0.keys(include_nested=True))

tmp0 = {
    "inputs": {
        "image": torch.rand(100, 28, 28),
        "mask": torch.randint(2, size=(100, 28, 28), dtype=torch.uint8)
    },
    "outputs": {"logits": torch.randn(100, 10)},
}
z0 = tensordict.TensorDict(tmp0, batch_size=[100])


# name
z0 = tensordict.TensorDict({}, batch_size=[3, 4], names=["a", None])
z0.names #["a", None]
z0.refine_names(..., "b") #["a", "b"]
z0.names = ["z", "y"]
z1 = z0.rename("m", "n") #["m", "n"] out-of-place
z2 = z1.rename(m="h")
z2.names #["h", "n"]


# nested tensordicts
tmp0 = {
    "inputs": {
        "image": torch.rand(100, 28, 28),
        "mask": torch.randint(2, size=(100, 28, 28), dtype=torch.uint8)
    },
    "outputs": {"logits": torch.randn(100, 10)},
}
z0 = tensordict.TensorDict(tmp0, batch_size=[100])
z0['inputs', 'image']
z0['inputs']['image']
z0["outputs", "probabilities"] = torch.sigmoid(z0['outputs','logits'])

z1 = tensordict.TensorDict({"b": torch.rand(3, 4, 5)}, batch_size=[3,4,5])
z2 = tensordict.TensorDict({"a": torch.rand(3,4), "nested":z1}, batch_size=[3,4])


# lazy evaluation
z0 = tensordict.TensorDict({"a":torch.rand(3,4), "b":torch.rand(3,4,5)}, batch_size=[3, 4])
z0_clone = z0.clone()
z1 = tensordict.LazyStackedTensorDict.lazy_stack([z0, z0_clone], dim=0)
z1.shape #(2,3,4)
z1['a'] *= 0
z0['a'] #all zero
z2 = z1.contiguous()


class DummyNet(torch.nn.Module):
    def __init__(self, dim_in:int):
        super().__init__()
        self.fc0 = torch.nn.Linear(dim_in, 1)
    def forward(self, x):
        logits = self.fc0(x)
        prob = torch.sigmoid(logits)
        return logits, prob
module = tensordict.nn.TensorDictModule(
    DummyNet(100),
    in_keys=["input"],
    out_keys=[("outputs", "logits"), ("outputs", "probabilities")],
)
z0 = tensordict.TensorDict({"input": torch.randn(32, 100)}, [32])
z1 = module(z0)
z1["outputs", "logits"]
z1["outputs", "probabilities"]
z2 = module(input=torch.randn(32, 100)) #output as a tensordict


class DummyNet01(torch.nn.Module):
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
class DummyNet02(torch.nn.Module):
    def forward(self, x, mask):
        return torch.softmax(x * mask, dim=1)
tmp0 = tensordict.nn.TensorDictModule(DummyNet01(), in_keys=[("input","x")], out_keys=[("intermediate","x")])
tmp1 = tensordict.nn.TensorDictModule(DummyNet02(), in_keys=[("intermediate","x"), ("input", "mask")], out_keys=[("output", "probabilities")])
module = tensordict.nn.TensorDictSequential(tmp0, tmp1)
tmp0 = {"input": {"x": torch.rand(32, 100), "mask": torch.randint(2, size=(32, 10))}}
z0 = tensordict.TensorDict(tmp0, batch_size=[32])
z1 = module(z0)
z1["intermediate", "x"]
z1["output", "probabilities"]




layer1 = torch.nn.Linear(3, 4)
layer2 = torch.nn.Linear(4, 4)
model = torch.nn.Sequential(layer1, layer2)
weights1 = tensordict.TensorDict(layer1.state_dict(), []).unflatten_keys(separator=".")
weights2 = tensordict.TensorDict(layer2.state_dict(), []).unflatten_keys(separator=".")
params = tensordict.nn.make_functional(model)
# params provided by make_functional match state_dict:
assert (params == tensordict.TensorDict({"0": weights1, "1": weights2}, [])).all()
x0 = torch.randn(10, 3)
x1 = model(x0, params=params)
params_stack = torch.stack([params, params], 0)
x2 = torch.vmap(model, (None, 0))(x0, params_stack) #shape: 2,10,4
