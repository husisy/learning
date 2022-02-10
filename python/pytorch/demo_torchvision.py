import numpy as np
import torch
import torchvision
from tqdm import tqdm


def topk_accuracy(predict, label, topk=(1,)):
    with torch.no_grad():
        tmp0 = predict.topk(max(topk), dim=1)[1]
        correct = (tmp0==label.view(-1,1))
        ret = [correct[:,:x].sum().item() for x in topk]
    return ret


def _demo_torchvision_model_ILSVRC2012_accuracy_main(model, val_folder, resize=256, crop=224, device='cuda'):
    device = torch.device(device)
    tmp0 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(val_folder, tmp0)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    model = model.to(device)

    model.eval()
    validation_top1_correct = 0
    validation_top5_correct = 0
    validation_total = 0
    with torch.no_grad():
        for data_i,label_i in tqdm(val_loader):
            data_i, label_i = data_i.to(device), label_i.to(device)
            predict = model(data_i)
            tmp0 = topk_accuracy(predict, label_i, topk=(1,5))
            validation_top1_correct += tmp0[0]
            validation_top5_correct += tmp0[1]
            validation_total += label_i.size()[0]
    top1_err = 1 - validation_top1_correct / validation_total
    top5_err = 1 - validation_top5_correct / validation_total
    return top1_err, top5_err


def demo_torchvision_model_ILSVRC2012_accuracy():
    model = torchvision.models.resnet18(pretrained=True)
    val_folder = '/media/hdd/pytorch_data/ILSVRC2012/val'
    top1_err,top5_err = _demo_torchvision_model_ILSVRC2012_accuracy_main(model, val_folder)
    print('resnet18: ', top1_err*100, top5_err*100) #30.242, 10.924

    model = torchvision.models.resnet50(pretrained=True)
    val_folder = '/media/hdd/pytorch_data/ILSVRC2012/val'
    top1_err,top5_err = _demo_torchvision_model_ILSVRC2012_accuracy_main(model, val_folder)
    print('resnet50: ', top1_err*100, top5_err*100) #23.87 7.138

    model = torchvision.models.inception_v3(pretrained=True)
    val_folder = '/media/hdd/pytorch_data/ILSVRC2012/val'
    top1_err,top5_err = _demo_torchvision_model_ILSVRC2012_accuracy_main(model, val_folder, resize=299, crop=299)
    print('inception_v3: ', top1_err*100, top5_err*100) #22.78, 6.47
    # https://pytorch.org/hub/pytorch_vision_inception_v3/
