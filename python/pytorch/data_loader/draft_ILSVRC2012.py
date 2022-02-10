import os
import time
import argparse
import PIL.Image
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", "The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors", UserWarning)

from utils import raw_to_lmdb, raw_to_hdf5, hdf5_to_raw
from utils import demo_hdf5_speed, demo_raw_filesystem_speed, demo_lmdb_speed, ImageHDF5Dataset


def demo_ilsvrc2012():
    imagedir = '/raid/pytorch_data/ILSVRC2012/train'
    lmdb_path = '/raid/pytorch_data/ILSVRC2012/ILSVRC2012_train_lmdb'
    hdf5_path = '/raid/pytorch_data/ILSVRC2012/ILSVRC2012_train.hdf5'
    image_path_list = [os.path.join(x,y) for x in sorted(os.listdir(imagedir)) for y in sorted(os.listdir(os.path.join(imagedir,x)))]
    key_list = [x.encode() for x in image_path_list]

    # raw_to_lmdb(image_path_list, lmdb_path, imagedir)
    # raw_to_hdf5(image_path_list, hdf5_path, imagedir)

    demo_lmdb_speed(lmdb_path, key_list)
    # dgx-station-ssd 40116it/s 598it/s 188it/s 148it/s
    # P720-hhd 47445it/s 80it/s 217it/s 41it/s

    demo_raw_filesystem_speed(image_path_list, imagedir=imagedir, use_cv2=False)
    # dgx-station-ssd 23107it/s 23195it/s 187it/s 188it/s
    # P720-hhd 30472it/s 124it/s 215it/s 54it/s
    demo_raw_filesystem_speed(image_path_list, imagedir=imagedir, use_cv2=True)

    demo_hdf5_speed(hdf5_path)
    # dgx-station-ssd 2805it/s 847it/s 176it/s 154it/s
    # P720-hdd 3057it/s 121it/s 199it/s 52it/s
