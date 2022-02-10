import os
import io
import lmdb
import h5py
import pickle
import PIL.Image
import numpy as np
from tqdm import tqdm
import torch
import cv2


def list_imagepath(imagedir):
    ret = [os.path.join(x,y) for x in sorted(os.listdir(imagedir)) for y in sorted(os.listdir(os.path.join(imagedir,x)))]
    return ret


def raw_to_lmdb(image_path_list, lmdb_path, imagedir=None):
    assert all(isinstance(x, str) for x in image_path_list)
    meta_file = lmdb_path + '_meta.pkl'
    if imagedir is None:
        hf_file = lambda *x: os.path.join(*x)
    else:
        hf_file = lambda *x: os.path.join(imagedir, *x)
    if os.path.exists(lmdb_path):
        print(f'directory "{lmdb_path}" already exists, skip it')
    else:
        with open(meta_file, 'wb') as fid:
            tmp0 = {'all_key': [x.encode() for x in image_path_list]}
            pickle.dump(tmp0, fid)
        map_size = int(sum(os.path.getsize(hf_file(x)) for x in image_path_list) * 1.1) #add 10% more space for lmdb
        print('map_size={}Byte={:.3}GB'.format(map_size, map_size/2**30))
        env = lmdb.open(lmdb_path, map_size=map_size)
        with env.begin(write=True) as txn:
            for x in tqdm(image_path_list):
                with open(hf_file(x), 'rb') as fid:
                    txn.put(x.encode(), fid.read())
        env.close()
    return lmdb_path, meta_file


def raw_to_hdf5(image_path_list, hdf5_path, imagedir=None, image_key='image'):
    assert all(isinstance(x, str) for x in image_path_list)
    assert hdf5_path.endswith('.hdf5')
    meta_file = hdf5_path[:-5] + '_hdf5_meta.pkl'
    if imagedir is None:
        hf_file = lambda *x: os.path.join(*x)
    else:
        hf_file = lambda *x: os.path.join(imagedir, *x)
    if os.path.exists(hdf5_path):
        print(f'file "{hdf5_path}" already exists, skip it')
    else:
        with open(meta_file, 'wb') as fid:
            pickle.dump({'image_path_list':image_path_list}, fid)
        N0 = len(image_path_list)
        with h5py.File(hdf5_path, 'w') as fid:
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            image_dset = fid.create_dataset(image_key, shape=(N0,), dtype=dt)
            for ind0,x in enumerate(tqdm(image_path_list)):
                with open(os.path.join(imagedir, x), 'rb') as image_fid:
                    image_dset[ind0] = np.frombuffer(image_fid.read(), dtype='uint8')
    return hdf5_path, meta_file


def hdf5_to_raw(hdf5_path, imagedir, meta_file=None, image_key='image'):
    assert hdf5_path.endswith('.hdf5')
    if meta_file is None:
        meta_file = hdf5_path[:-5] + '_hdf5_meta.pkl'
    hf_file = lambda *x: os.path.join(imagedir, *x)
    with open(meta_file, 'rb') as fid:
        image_path_list = pickle.load(fid)['image_path_list']
    for x in {x.split(os.sep,1)[0] for x in image_path_list}:
        if not os.path.exists(hf_file(x)):
            os.makedirs(hf_file(x))
    with h5py.File(hdf5_path, 'r') as fid:
        image_dset = fid[image_key]
        for ind0 in tqdm(range(len(image_path_list))):
            image_path = os.path.join(imagedir, image_path_list[ind0])
            with open(image_path, 'wb') as fid:
                fid.write(io.BytesIO(image_dset[ind0]).read())


class ImageHDF5Dataset(torch.utils.data.Dataset):
    '''usage:
    >>> myset = ImageHDF5Dataset('tbd00.hdf5')
    >>> myset[233]
    >>> len(myset)
    '''
    def __init__(self, hdf5_path, label_or_strToID=None, meta_file=None, image_key='image', transform=None, target_transform=None):
        self.hdf5_path = hdf5_path
        self.hdf5_fid = None
        self.image_dset = None
        self.image_key = image_key
        self.transform = transform
        self.target_transform = target_transform

        if meta_file is None:
            meta_file = hdf5_path[:-5] + '_hdf5_meta.pkl'
        with open(meta_file, 'rb') as fid:
            self.image_path_list = pickle.load(fid)['image_path_list']

        if isinstance(label_or_strToID, dict):
            self.wnid_to_index = label_or_strToID
            tmp0 = [self.wnid_to_index[x.split(os.sep,1)[0]] for x in self.image_path_list]
            self.label_np = np.array(tmp0)
        elif label_or_strToID is None:
            tmp0 = sorted({x.split(os.sep,1)[0] for x in self.image_path_list})
            self.wnid_to_index = {y:x for x,y in enumerate(tmp0)}
            tmp0 = [self.wnid_to_index[x.split(os.sep,1)[0]] for x in self.image_path_list]
            self.label_np = np.array(tmp0)
        else:
            assert len(label_or_strToID)==len(self.image_path_list)
            tmp0 = {(x,y.split(os.sep,1)[0]) for x,y in zip(label_or_strToID,self.image_path_list)}
            assert len(label_or_strToID)==len(tmp0)
            self.wnid_to_index = {y:x for x,y in tmp0}
            self.label_np = np.array(label_or_strToID)

    def __getitem__(self, index):
        if self.image_dset is None:
            self.hdf5_fid = h5py.File(self.hdf5_path, 'r')
            self.image_dset = self.hdf5_fid[self.image_key]
        image_pil = PIL.Image.open(io.BytesIO(self.image_dset[index]))
        if image_pil.mode!='RGB':
            image_pil = image_pil.convert('RGB')
        if self.transform is not None:
            image_pil = self.transform(image_pil)
        target = self.label_np[index]
        if self.target_transform is not None:
            target = target_transform(target)
        return image_pil,target

    def __len__(self):
        return self.label_np.size


def demo_hdf5_speed(hdf5_path, N0=10000, image_key='image', task=(0,1,2,3), use_cv2=False):
    if 0 in task:
        print('[hdf5] linear read binary')
        with h5py.File(hdf5_path, 'r') as fid:
            image_dset = fid[image_key]
            np0 = np.arange(len(image_dset))[:N0]
            for ind0 in tqdm(np0):
                _ = io.BytesIO(image_dset[ind0])

    if 1 in task:
        print('[hdf5] random read binary')
        with h5py.File(hdf5_path, 'r') as fid:
            image_dset = fid[image_key]
            np0 = np.random.permutation(len(image_dset))[:N0]
            for ind0 in tqdm(np0):
                _ = io.BytesIO(image_dset[ind0])

    if 2 in task:
        print('[hdf5] linear read binary and decode')
        with h5py.File(hdf5_path, 'r') as fid:
            image_dset = fid[image_key]
            np0 = np.arange(len(image_dset))[:N0]
            for ind0 in tqdm(np0):
                _ = np.asarray(PIL.Image.open(io.BytesIO(image_dset[ind0])))

    if 3 in task:
        print('[hdf5] random read binary and decode')
        with h5py.File(hdf5_path, 'r') as fid:
            image_dset = fid[image_key]
            np0 = np.random.permutation(len(image_dset))[:N0]
            for ind0 in tqdm(np0):
                _ = np.asarray(PIL.Image.open(io.BytesIO(image_dset[ind0])))


def demo_raw_filesystem_speed(image_path_list, imagedir=None, N0=10000, task=(0,1,2,3), use_cv2=False):
    if imagedir is None:
        hf_file = lambda *x: os.path.join(*x)
    else:
        hf_file = lambda *x: os.path.join(imagedir, *x)

    if 0 in task:
        print('[raw-filesystem] linear read binary')
        np0 = np.arange(len(image_path_list))[:N0]
        for ind0 in tqdm(np0):
            with open(hf_file(image_path_list[ind0]), 'rb') as fid:
                _ = io.BytesIO(fid.read())

    if 1 in task:
        print('[raw-filesystem] random read binary')
        np0 = np.random.permutation(len(image_path_list))[:N0]
        for ind0 in tqdm(np0):
            with open(hf_file(image_path_list[ind0]), 'rb') as fid:
                _ = io.BytesIO(fid.read())

    if 2 in task:
        print('[raw-filesystem] linear read binary and decode')
        np0 = np.arange(len(image_path_list))[:N0]
        for ind0 in tqdm(np0):
            image_path = hf_file(image_path_list[ind0])
            if use_cv2:
                _ = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                with PIL.Image.open(image_path) as image_pil:
                    _ = np.asarray(image_pil)

    if 3 in task:
        print('[raw-filesystem] random read binary and decode')
        np0 = np.random.permutation(len(image_path_list))[:N0]
        for ind0 in tqdm(np0):
            image_path = hf_file(image_path_list[ind0])
            if use_cv2:
                _ = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                with PIL.Image.open(image_path) as image_pil:
                    _ = np.asarray(image_pil)


def demo_lmdb_speed(lmdb_path, key_list, N0=10000, task=(0,1,2,3)):
    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

    if 0 in task:
        print('[lmdb] linear read binary')
        index_list = [key_list[x] for x in np.arange(len(key_list))[:N0]]
        with env.begin(write=False) as txn:
            for x in tqdm(index_list):
                _ = io.BytesIO(txn.get(x))

    if 1 in task:
        print('[lmdb] random read binary')
        index_list = [key_list[x] for x in np.random.permutation(len(key_list))[:N0]]
        with env.begin(write=False) as txn:
            for x in tqdm(index_list):
                _ = io.BytesIO(txn.get(x))

    if 2 in task:
        print('[lmdb] linear read binary and decode')
        index_list = [key_list[x] for x in np.arange(len(key_list))[:N0]]
        with env.begin(write=False) as txn:
            for x in tqdm(index_list):
                _ = np.asarray(PIL.Image.open(io.BytesIO(txn.get(x))))

    if 3 in task:
        print('[lmdb] random read binary and decode')
        index_list = [key_list[x] for x in np.random.permutation(len(key_list))[:N0]]
        with env.begin(write=False) as txn:
            for x in tqdm(index_list):
                _ = np.asarray(PIL.Image.open(io.BytesIO(txn.get(x))))
    env.close()
