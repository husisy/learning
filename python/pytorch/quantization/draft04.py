from typing import Type, Dict, Any, Tuple, Iterable
import copy
# import torch.fx as fx
import torch
# import torch.nn as nn
import torch.fx
import torchvision
import time

class WrappedBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        return self.mod(x)

class MyModel00(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.nested = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 1, 1),
        )
        self.wrapped = WrappedBatchNorm()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.nested(x)
        x = self.wrapped(x)
        return x

model = MyModel00()
model.eval()

traced_model = torch.fx.symbolic_trace(model)
print(traced_model.graph)


def fuse_conv_bn_eval(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    assert (not conv.training) and (not bn.training), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: torch.fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    # The first step of most FX passes is to symbolically trace our model to
    # obtain a `GraphModule`. This is a representation of our original model
    # that is functionally identical to our original model, except that we now
    # also have a graph representation of our forward pass.
    fx_model: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    # The primary representation for working with FX are the `Graph` and the
    # `Node`. Each `GraphModule` has a `Graph` associated with it - this
    # `Graph` is also what generates `GraphModule.code`.
    # The `Graph` itself is represented as a list of `Node` objects. Thus, to
    # iterate through all of the operations in our graph, we iterate over each
    # `Node` in our `Graph`.
    for node in fx_model.graph.nodes:
        # The FX IR contains several types of nodes, which generally represent
        # call sites to modules, functions, or methods. The type of node is
        # determined by `Node.op`.
        if node.op != 'call_module': # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        if type(modules[node.target]) is torch.nn.BatchNorm2d and type(modules[node.args[0].target]) is torch.nn.Conv2d:
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue
            conv = modules[node.args[0].target]
            bn = modules[node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)
            replace_node_module(node.args[0], modules, fused_conv)
            # As we've folded the batch nor into the conv, we need to replace all uses
            # of the batch norm with the conv.
            node.replace_all_uses_with(node.args[0])
            # Now that all uses of the batch norm have been replaced, we can
            # safely remove the batch norm.
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    # After we've modified our graph, we need to recompile our graph in order
    # to keep the generated code in sync.
    fx_model.recompile()
    return fx_model


fused_model = fuse(model)
print(fused_model.code)
inp = torch.randn(5, 1, 1, 1)
torch.testing.assert_allclose(fused_model(inp), model(inp))



# resnet18
model_resnet18 = torchvision.models.resnet18()
model_resnet18.eval()

inp = torch.randn(10, 3, 224, 224)
output = model_resnet18(inp)

def benchmark(model, iters=20):
    for _ in range(10):
        model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str(time.time()-begin)

model_resnet18_fused = fuse(model_resnet18)
print("Unfused time: ", benchmark(model_resnet18))
print("Fused time: ", benchmark(model_resnet18_fused))

model_resnet18_fused_jit = torch.jit.script(model_resnet18_fused)
print("jit time: ", benchmark(model_resnet18_fused_jit))
