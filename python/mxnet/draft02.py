# finetune https://mxnet.apache.org/api/python/docs/tutorials/getting-started/crash-course/5-predict.html
import mxnet as mx
import numpy as np
# import matplotlib.pyplot as plt
# from mxnet.gluon.model_zoo import vision as models
# from mxnet.gluon.utils import download
# from mxnet import image

filename = mx.gluon.utils.download('http://data.mxnet.io/models/imagenet/synset.txt')
with open(filename, 'r') as fid:
    id_to_label = dict(enumerate([x.strip() for x in fid]))

url = ('https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/'
    'Golden_Retriever_medium-to-light-coat.jpg/365px-Golden_Retriever_medium-to-light-coat.jpg')
filename = mx.gluon.utils.download(url)
mx0 = mx.image.imread(filename) #(mx,uint8,(480,365,3))
tmp0 = mx.image.resize_short(mx0, 256) #resize the short edge into 256 pixels
mx1,_ = mx.image.center_crop(tmp0, (224,224))


def resnet50_transform(data):
    # data: channel-last
    data = data.transpose((2,0,1)).expand_dims(axis=0).astype('float32') / 255
    mean = mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    std = mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    ret = (data - mean) / std
    return ret

net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True)
probability = net(resnet50_transform(mx1)).softmax()[0].asnumpy()
index_top5 = np.argsort(probability)[-5:][::-1]
for x in index_top5:
    print('{:.4}: {}'.format(probability[x], id_to_label[x]))
