import torch


## dynamic quantization
class MyModel00(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

model_fp32 = MyModel00()
model_int8 = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)
len([x for x in model_int8.parameters() if x.requires_grad]) #0
torch0 = torch.randn(6, 4)
torch1 = model_int8(torch0)



# static quantization
class MyModel01(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub() #convert float point tensors to quantized
        self.conv = torch.nn.Conv2d(1, 1, 1) #(in_channel,out_channel,kernel_size)
        self.relu = torch.nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub() #from quantized to float point

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

model_fp32 = MyModel01()
model_fp32.eval() #must be eval mode for static quantization
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm') #fbgemm(x86) qnnpack(arm)
# Common fusions: conv+relu, conv+bn+relu
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
# inserts observers in the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)
# 1. quantizes the weights
# 2. computes and stores the scale and bias value to be used with each activation tensor
# 3. replaces key operators with quantized implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)

res = model_int8(input_fp32) #calculate in int8


## quantization aware training
class MyModel02(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

model_fp32 = MyModel02()
model_fp32.train() #must be train mode for QAT
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'bn', 'relu']])
# inserts observers and fake_quants in the model that will observe weight and activation tensors during calibration
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)

# training_loop(model_fp32_prepared)....

# 1. quantizes the weights
# 2. computes and stores the scale and bias value to be used with each activation tensor
# 3. fuses modules where appropriate,
# 4. replaces key operators with quantized implementations.
model_fp32_prepared.eval()
model_int8 = torch.quantization.convert(model_fp32_prepared)

res = model_int8(input_fp32) #calculate in int8
