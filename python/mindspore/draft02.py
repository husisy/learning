import os
import numpy as np
import mindspore as ms

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='GPU')


# dataset
ILSVRC2012_ROOT = '/home/xupx/data'
# ILSVRC2012_ROOT = '/mnt/ILSVRC2012'

train_dir = os.path.join(ILSVRC2012_ROOT, 'train')
test_dir = os.path.join(ILSVRC2012_ROOT, 'val')

datadir = train_dir
world_size = 1
rank = 0
image_size = 224

# TODO read byte data mannually
ds = ms.dataset.engine.ImageFolderDataset(datadir, num_parallel_workers=8, shuffle=True,
                            num_shards=world_size, shard_id=rank)
mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
