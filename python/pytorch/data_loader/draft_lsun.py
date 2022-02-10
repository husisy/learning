import os
import io
import h5py
import PIL.Image
import lmdb
import pickle
import string
import shutil
import numpy as np
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import demo_hdf5_speed, demo_raw_filesystem_speed, demo_lmdb_speed

def copy_LSUN_pytorch_cache(root, target_dir='.', classes=None, copy=True):
    '''
    generate cache used by torchvision.datasets.LSUN, but lmdb is really sloooooow
    '''
    all_class = []
    tmp0 = ['conference_room', 'classroom', 'bedroom', 'living_room', 'bridge',
             'dining_room', 'church_outdoor', 'tower', 'restaurant', 'kitchen']
    all_class = [(x+'_train') for x in tmp0] + [(x+'_val') for x in tmp0] + ['test']
    if classes is None:
        classes = all_class
    else:
        assert set(classes) <= set(all_class)
    hf_file = lambda *x: os.path.join(root, *x)
    pytorch_cache_dir = 'pytorch_cache'
    if not os.path.exists(hf_file(pytorch_cache_dir)):
        os.makedirs(hf_file(pytorch_cache_dir))
    for class_i in classes:
        cache_file = hf_file(pytorch_cache_dir, f'{class_i}.pkl')
        if os.path.exists(cache_file):
            print(f'cache file "{cache_file}" already exists, skip it')
        else:
            print(f'start to generate cache file "{cache_file}"')
            env = lmdb.open(hf_file(f'{class_i}_lmdb'), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                num_total = txn.stat()['entries']
                keys = [key for key, _ in tqdm(txn.cursor(), total=num_total)]
            with open(cache_file, 'wb') as fid:
                pickle.dump(keys, fid)
            env.close()
        if copy:
            tmp0 = '_cache_' + ''.join(c for c in f'{root}{class_i}_lmdb' if c in string.ascii_letters)
            target_file = os.path.join(target_dir, tmp0)
            if not os.path.exists(target_file):
                shutil.copyfile(cache_file, target_file)


def extract_LSUN_to_raw(lmdb_path, imagedir):
    if os.path.exists(imagedir):
        print(f'directory "{imagedir}" already exists, skip it')
        return
    else:
        os.makedirs(imagedir)
    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    hf_file = lambda *x: os.path.join(imagedir, *x)
    with env.begin(write=False) as txn:
        num_total = txn.stat()['entries']
        for key,value in tqdm(txn.cursor(), total=num_total):
            with open(hf_file(key.decode()+'.jpg'), 'wb') as fid:
                fid.write(value)
    env.close()


def LSUN_lmdb_to_hdf5(lmdb_path, hdf5_path):
    if os.path.exists(hdf5_path):
        print(f'file "{hdf5_path}" already exists, skip it')
        return
    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        num_total = txn.stat()['entries']
        with h5py.File(hdf5_path, 'w') as hdf5_fid:
            dt = h5py.special_dtype(vlen=str)
            image_path_dset = hdf5_fid.create_dataset('image_path', shape=(num_total,), dtype=dt)
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            image_dset = hdf5_fid.create_dataset('image', shape=(num_total,), dtype=dt)

            for ind0,(key,value) in enumerate(tqdm(txn.cursor(), total=num_total)):
                image_path_dset[ind0] = key.decode()
                image_dset[ind0] = np.frombuffer(value, dtype='uint8')
    env.close()


def demo_LSUN():
    copy_LSUN_pytorch_cache('/media/hdd/pytorch_data/LSUN', classes=['bedroom_train'])
    dataset = torchvision.datasets.LSUN('/media/hdd/pytorch_data/LSUN', classes=['bedroom_train'])


def demo_LSUN_speed():
    lmdb_path = '/media/hdd/pytorch_data/LSUN/bedroom_train_lmdb'
    imagedir = '/media/hdd/pytorch_data/LSUN/raw/train/bedroom'
    hdf5_path = '/media/hdd/pytorch_data/LSUN/raw/bedroom_train.hdf5'
    with h5py.File(hdf5_path, 'r') as hdf5_fid:
        image_path_list = hdf5_fid['image_path'][...].tolist()
    key_list = [x[:-4].encode() for x in image_path_list]
    demo_lmdb_speed(lmdb_path, key_list)
    # 75163it/s, 390it/s 747it/s 180it/s

    demo_raw_filesystem_speed(image_path_list, imagedir=imagedir)
    # 58563it/s 129it/s 713it/s 78it/s

    demo_hdf5_speed(hdf5_path)
    # 5253it/s 192it/s 636it/s 113it/s
