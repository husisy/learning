# https://pytorch.org/docs/stable/onnx.html
import os
import numpy as np
import scipy.special
import onnx
import onnxruntime as ort
import torch
import torchvision

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

np_rng = np.random.default_rng()
np0 = np_rng.normal(size=(10,3,224,224)).astype(np.float32)
torch0 = torch.tensor(np0, dtype=torch.float32, device='cuda')
torch_model = torchvision.models.alexnet(pretrained=True).cuda()
torch_model.eval()
with torch.no_grad():
    ret_ = scipy.special.softmax(torch_model(torch0).cpu().numpy())

onnx_filepath = hf_file('alexnet.onnx')

input_names = ['actual_input_1'] + [f'learned_{i}' for i in range(16)]
output_names = ['output1']
torch.onnx.export(torch_model, torch0, onnx_filepath, verbose=True,
    input_names=input_names, output_names=output_names)


onnx_model = onnx.load(onnx_filepath)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)


ort_session = ort.InferenceSession('alexnet.onnx')
ret0 = scipy.special.softmax(ort_session.run(None, {'actual_input_1': np0})[0]) #unfortunately differ from ret_
assert hfe(ret_, ret0) < 0.005
