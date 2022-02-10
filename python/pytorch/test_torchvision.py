import numpy as np
import torch
import torchvision

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def test_transforms_Normalize():
    np_mean = np.random.rand(3).astype(np.float32)
    np_std = np.random.rand(3).astype(np.float32) + 1

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(np_mean, np_std),
    ])

    np0 = np.random.rand(23, 23, 3).astype(np.float32)
    ret_ = np.transpose((np0 - np_mean)/np_std, (2,0,1))
    torch0 = transform(np0)
    assert hfe(ret_, torch0.numpy()) < 1e-4

    np0 = np.random.randint(256, size=(23,23,3), dtype=np.uint8)
    ret_ = np.transpose((np0.astype(np.float32)/255 - np_mean)/np_std, (2,0,1))
    torch0 = transform(np0)
    assert hfe(ret_, torch0.numpy()) < 1e-4
