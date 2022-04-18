import os
import tempfile
import itertools
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

def get_model_file_size(model):
    state_dict = model.state_dict()
    # num_parameter = sum(x.numel() for x in state_dict.values())

    # tmp0 = sorted([(str(x.dtype).rsplit('.',1)[1], x.numel()) for x in state_dict.values()], key=lambda x:x[0])
    # dtype_to_byte = {'float32':4, 'float64':8, 'int32':4, 'int64':8}
    # size = sum([dtype_to_byte[x0]*sum(y[1] for y in x1) for x0,x1 in itertools(tmp0, key=lambda x:x[0])])

    z0 = tempfile.TemporaryDirectory()
    filepath = os.path.join(z0.name, 'model.pt')
    torch.save(state_dict, filepath)
    size_in_MB = os.path.getsize(filepath)/2**20
    os.remove(filepath)
    z0.cleanup()
    return size_in_MB

model = torchvision.models.mobilenet_v2(pretrained=True)
model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

backend = "qnnpack" #qnnpack(arm) fbgemm(x86)
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
tmp0 = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(tmp0, inplace=False)

print('size: {:.4} MB'.format(get_model_file_size(model)))
print('[quantized] size: {:.4} MB'.format(get_model_file_size(model_quantized)))
print('[dynamic_quantized] size: {:.4} MB'.format(get_model_file_size(model_dynamic_quantized)))
print('[static_quantized] size: {:.4} MB'.format(get_model_file_size(model_static_quantized)))
