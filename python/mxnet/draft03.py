# DONE use pytorch to generate cifar10 data in filesystem
# DONE use mxnet to generate recordIO from filesystem
# DONE read recordIO to match pytorch.data
# TODO unittest for recordIO instead of im2rec.py, rewrite im2rec.py, not use cv2
# TODO nvidia-DALI to read recordIO
import os
import numpy as np
import mxnet as mx
import nvidia.dali

def generate_cifar10_dataset(datadir='data/cifar10'):
    import torchvision
    trainset = torchvision.datasets.CIFAR10(root='~/pytorch_data', train=True)
    id_to_label = {y:x for x,y in trainset.class_to_idx.items()}
    id_to_image = {y:[z0 for z0,z1 in trainset if z1==x] for x,y in id_to_label.items()}
    label_list = [x[0] for x in sorted(trainset.class_to_idx.items(), key=lambda x:x[1])]
    np_data = [np.stack([np.asarray(y) for y in id_to_image[x]]) for x in label_list]
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        for label,image_list in id_to_image.items():
            os.makedirs(os.path.join(datadir, label))
            for ind0,image in enumerate(image_list):
                image.save(os.path.join(datadir, label, f'{ind0}.PNG'))
    return np_data,label_list


# conda activate cuda102_cv2
# python im2rec.py ./cifar10 ../data/cifar10/ --recursive --list --num-thread 8
'''
airplane 0
automobile 1
bird 2
cat 3
deer 4
dog 5
frog 6
horse 7
ship 8
truck 9
'''
# python im2rec.py ./cifar10 ../data/cifar10/ --recursive --pass-through --pack-label --num-thread 8

np_data,label_list = generate_cifar10_dataset('../data/cifar10')

trainset = mx.io.ImageRecordIter(path_imgrec='cifar10.rec', path_imgidx='cifar10.idx',
        data_shape=(3,32,32), batch_size=5, shuffle=False)
# tmp0 = next(iter(trainset))
# data = tmp0.data[0] #(mx,float32,(5,3,32,32)) cpu_pinned
# label = tmp0.label[0] #(mx,float32,(5,)) cpu_pinned

with open('cifar10.lst') as fid:
    tmp0 = [x.strip().split()[-1].split('/') for x in fid]
    z0 = [(x,int(y.split('.')[0])) for x,y in tmp0]
tmp0 = [y.astype(np.uint8) for x in trainset for y in x.data[0].asnumpy()]
tmp1 = np.stack([x[0] for x in sorted(zip(tmp0,z0), key=lambda x:x[1])]).transpose(0,2,3,1)
mx_data = list(tmp1.reshape(-1,5000,*tmp1.shape[1:]))
assert all(np.all(x==y) for x,y in zip(np_data,mx_data))


prefix = 'cifar10'
datadir = '../data/cifar10'
# list True
# exts = [jpeg jpg png]
# chunks 1
# train-ratio 1
# test-ratio 0
# recursive True
# no-shuffle True
# pass-through True
# resize 0
# center-crop False
# quality 95
# num-thread 8

# nvidia.dali.pipeline.Pipeline
# nvidia.dali.ops
# nvidia.dali.types

class RecordIOPipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self, rec_filepath, idx_filepath, batch_size, num_threads, device_id):
        super(RecordIOPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = nvidia.dali.ops.MXNetReader(path=[rec_filepath], index_path=[idx_filepath])

        self.decode = nvidia.dali.ops.ImageDecoder(device='cpu', output_type=nvidia.dali.types.RGB)

    def define_graph(self):
        image, labels = self.input(name='Reader')
        images = self.decode(image)
        return (image, labels)


pipe = RecordIOPipeline('cifar10.rec', 'cifar10.idx', 64, 1, 0)
pipe.build()
image,label = pipe.run()
image_list = [image.at(x) for x in range(len(image))]
# TODO fail error record
