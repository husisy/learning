# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
import os
import time
import tempfile
import numpy as np
import torch
import torchvision
import torch.quantization
import torch.nn.functional as F
from tqdm import tqdm
# from tqdm.notebook import tqdm

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            torch.nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            torch.nn.ReLU(inplace=False)
        )


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = torch.nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(torch.nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = torch.nn.Sequential(*features)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # building classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == torch.nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        ret = fmtstr.format(name=self.name, val=self.val, avg=self.avg)
        return ret


def topk_accuracy(logits, label, topk=(1,)):
    with torch.no_grad():
        prediction = torch.topk(logits, max(topk), dim=1)[1].cpu().numpy()
    tmp0 = np.cumsum(prediction==(label.cpu().numpy()[:,np.newaxis]), axis=1)
    ret = np.mean(tmp0[:,[x-1 for x in topk]]>=1, axis=0)
    return ret

def evaluate(model, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for _,(image,target) in zip(tqdm(range(neval_batches)), data_loader):
            output = model(image)
            acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
            top1.update(acc1, image.size(0))
            top5.update(acc5, image.size(0))
    print(f'acc1={top1.sum}/{top1.count}={100*top1.avg:.2f}; acc5={top5.sum}/{top5.count}={100*top5.avg:.2f}')


def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def get_model_file_size(model):
    state_dict = model.state_dict()
    z0 = tempfile.TemporaryDirectory()
    filepath = os.path.join(z0.name, 'model.pt')
    torch.save(state_dict, filepath)
    size_in_MB = os.path.getsize(filepath)/2**20
    os.remove(filepath)
    z0.cleanup()
    return size_in_MB

def prepare_data_loaders(data_dir, train_batch_size, eval_batch_size):
    normalize = torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tmp0 = torchvision.transforms.Compose([
                   torchvision.transforms.RandomResizedCrop(224),
                   torchvision.transforms.RandomHorizontalFlip(),
                   torchvision.transforms.ToTensor(),
                   normalize,
    ])
    ds_train = torchvision.datasets.ImageNet(data_dir, split="train", transform=tmp0)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=train_batch_size, shuffle=True)

    tmp0 = torchvision.transforms.Compose([
                  torchvision.transforms.Resize(256),
                  torchvision.transforms.CenterCrop(224),
                  torchvision.transforms.ToTensor(),
                  normalize,
    ])
    ds_val = torchvision.datasets.ImageNet(data_dir, split="val", transform=tmp0)
    dl_test = torch.utils.data.DataLoader(ds_val, batch_size=eval_batch_size, shuffle=False)
    return dl_train, dl_test

# wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth -O mobilenet_pretrained_float.pth
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
dl_train, dl_test = prepare_data_loaders('/opt/hdd/pytorch_data/ILSVRC2012', 30, 50)



## model_fp32
model_fp32 = load_model(float_model_file).to('cpu')

# Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
# while also improving numerical accuracy. While this can be used with any model, this is
# especially common with quantized models.

print('\n Inverted Residual Block: Before fusion \n\n', model_fp32.features[1].conv)
model_fp32.eval()

model_fp32.fuse_model() #fusion of Conv+BN+Relu and Conv+Relu
print('\n Inverted Residual Block: After fusion\n\n',model_fp32.features[1].conv)

print('model_fp32(MB):', get_model_file_size(model_fp32)) #13.99MB

evaluate(model_fp32, dl_test, neval_batches=10) #20minutes for neval_batches=1000
torch.jit.save(torch.jit.script(model_fp32), scripted_float_model_file)


## post-training static quantization (PTQ), per-layer quantization
num_calibration_batches = 32
model_ptq = load_model(float_model_file).to('cpu')
model_ptq.eval()
model_ptq.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
model_ptq.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model_ptq, inplace=True)

# Calibration
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', model_ptq.features[1].conv)
evaluate(model_ptq, dl_train, neval_batches=32) #40 seconds

# Convert to quantized model
torch.quantization.convert(model_ptq, inplace=True)
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',model_ptq.features[1].conv)

print('model_ptq(MB):', get_model_file_size(model_ptq)) #3.627281 MB

evaluate(model_ptq, dl_test, neval_batches=10) #10minutes


## post-training static quantization, per-channel quantization
model_ptq_per_channel = load_model(float_model_file)
model_ptq_per_channel.eval()
model_ptq_per_channel.fuse_model()
model_ptq_per_channel.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model_ptq_per_channel.qconfig)

torch.quantization.prepare(model_ptq_per_channel, inplace=True)
evaluate(model_ptq_per_channel, dl_train, neval_batches=32)
torch.quantization.convert(model_ptq_per_channel, inplace=True)
evaluate(model_ptq_per_channel, dl_test, neval_batches=10)
torch.jit.save(torch.jit.script(model_ptq_per_channel), scripted_quantized_model_file)



## Quantization aware training
qat_model = load_model(float_model_file)
qat_model.eval()
qat_model.fuse_model()
qat_model.train()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

torch.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
ntrain_batches = 20
for ind_epoch in range(8):
    qat_model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', ':1.5f')
    device = torch.device('cpu')
    with tqdm(range(ntrain_batches), desc=f'[epoch={ind_epoch}]') as pbar:
        for _,(data,label) in zip(pbar, dl_train):
            data, label = data.to(device), label.to(device)
            output = qat_model(data)
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, acc5 = topk_accuracy(output, label, topk=(1, 5))
            top1.update(acc1, data.size(0))
            top5.update(acc5, data.size(0))
            avgloss.update(loss, data.size(0))
            pbar.set_postfix(loss=f'{avgloss.avg:.5f}', acc1=f'{100*top1.avg:.2f}', acc5=f'{100*top5.avg:.2f}')

    if ind_epoch > 3: # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if ind_epoch > 2: # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    evaluate(quantized_model, dl_test, neval_batches=10)


## speedup from quantization
def mytimeit(hf0, num_repeat=10, is_inference=True):
    torch.set_num_threads(1) #quantized models run single threaded
    t_list = np.zeros(num_repeat+2, dtype=np.float64)
    for ind0 in range(num_repeat+2):
        t0 = time.time()
        if is_inference:
            with torch.no_grad():
                hf0()
        else:
            hf0()
        t_list[ind0] = time.time() - t0
    ret = (t_list.sum() - t_list.max() - t_list.min()) / num_repeat
    return ret

torch0 = next(iter(dl_test))[0]

model_fp32 = torch.jit.load(scripted_float_model_file)
model_fp32.eval()
tmp0 = mytimeit(lambda: model_fp32(torch0)) / len(torch0)
print(f'model_fp32: {tmp0*1000:.3f} ms per image') #41ms

model_quantized = torch.jit.load(scripted_quantized_model_file)
model_quantized.eval()
tmp0 = mytimeit(lambda: model_quantized(torch0)) / len(torch0)
print(f'model_quantized: {tmp0*1000:.3f} ms per image') #39ms
