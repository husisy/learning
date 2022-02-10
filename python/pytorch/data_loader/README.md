# data loader

1. link
   * [pytorch-blog/efficient IO](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/)
   * [github/pytorch-imagenet-wds](https://github.com/tmbdev/pytorch-imagenet-wds)
   * [wds RFC](https://github.com/pytorch/pytorch/issues/38419)
   * [pytorch RFC tracker](https://github.com/pytorch/pytorch/issues/41292)
   * [nvidia/DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_recordio.html)
   * [github/tfrecord](https://github.com/vahidk/tfrecord)

## hdf5压缩图片存储解决方案

1. link
   * [github-issue/save jpeg images in hd5py](https://github.com/h5py/h5py/issues/745)
   * [pytorch-forum/how to speed up the data loader](https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740)
   * [pytorch-forum/recommend the way to load larger h5 files](https://discuss.pytorch.org/t/recommend-the-way-to-load-larger-h5-files/32993)
   * [pytorch-dataloader-hdf5](https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/30)

| - | - | `linear` | `random` | `random+decode` | `random+4decode` |
| :-: | :-: | :-: | :-: | :-: | :-: |
| `dgx-station` | `raw` | `18049` | `19166` | `170` | `535` |
| `dgx-station` | `hdf5` | `2696` | `2739` | `158` | `517` |
| `P720-ssd` | `raw` | `23081` | `22822` | `192` | `552` |
| `P720-ssd` | `hdf5` | `2943` | `2764` | `180` | `528` |
| `P720-hdd` | `raw` | `377` | `825` | `154` | `461` |
| `P720-hdd` | `hdf5` | `2982` | `2989` | `179` | `530` |

结论

1. hdd就足够供应4GPU的训练速度（4GPU在ImageNet+resnet50任务上速度大约为`3200it/s`）
2. hdd上`hdf5`会显著由于`raw`

## ILSVRC2012

使用ILSVRC2012数据集中的`train`图片进行速度测试，总共包含`1281167`张图片

| - | - | linear | random | linear-decode | random-decode |
| :-: | :-: | :-: | :-: | :-: | :-: |
| `dgx-ssd` | `raw` | `x` | `x` | `x` | `x` |
| `dgx-ssd` | `lmdb` | `x` | `x` | `x` | `x` |
| `dgx-ssd` | `hdf5` | `x` | `x` | `x` | `x` |

## LSUN

使用LSUN数据集中的`bedroom-train`图片进行速度测试，总共包含`3033042`张图片

| - | linear | random | linear-decode | random-decode |
| :-: | :-: | :-: | :-: | :-: |
| `P720-hdd-lmdb` | `75163` | `390` | `747` | `180` |
| `P720-hdd-raw` | `58563` | `129` | `713` | `78` |
| `P720-hdd-hdf5` | `5253` | `192` | `636` | `113` |
| `dgx-ssd-lmdb` | `78386` | `104898` | `640` | `636` |
| `dgx-ssd-raw` | `34411` | `38366` | `621` | `616` |
| `dgx-ssd-hdf5` | `3886` | `3406` | `532` | `523` |

1. 文件大小
   * `lmdb`文件大小为`51GB`
   * `hdf5`文件大小为`48GB`
   * `raw`文件大小为`51GB`
2. 两次运行取最优那一次运行结果
