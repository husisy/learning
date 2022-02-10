import os
import math
import struct
import random

hf_data = lambda *x: os.path.join('data', *x)

def uint32_to_bytes(x):
    assert 0 <= x < 2**32
    ret = bytes([
        (x & 0xff000000) >> 24,
        (x & 0xff0000) >> 16,
        (x & 0xff00) >> 8,
        (x & 0xff),
    ])
    return ret


def uint_to_bytes(x, length=None):
    assert x>=0
    if (length is None) and (x==0):
        return bytes([0])
    if length is not None:
        assert x < 2**(8*length)
    else:
        length = math.ceil(math.log2(x+1)/8)
    if length==1:
        return bytes([x])
    tmp0 = list(reversed(range(0,8*length,8)))
    tmp1 = [2**(y+8)-2**y for y in tmp0]
    ret = bytes([(x&z)>>y for y,z in zip(tmp0,tmp1)])
    return ret


def demo_pack_unpack():
    # >: big-endian
    # I: 4-byte unsigned integer
    # H: 2-byte unsigned integer
    x = random.randint(0, 2**32-1)
    assert uint32_to_bytes(x) == uint_to_bytes(x,4)
    assert struct.pack('>I', x) == uint_to_bytes(x,4)
    assert struct.unpack('>I', struct.pack('>I', x))[0]==x

    x0 = random.randint(0, 2**32-1)
    x1 = random.randint(0, 2**16-1)
    ret_ = struct.pack('>IH', x0, x1)
    ret = uint_to_bytes(x0, 4) + uint_to_bytes(x1, 2)
    assert ret_==ret
    assert (x0,x1)==struct.unpack('>IH', ret_)


def detect_bmp_info(filepath):
    # see https://www.liaoxuefeng.com/wiki/1016959663602400/1017685387246080
    assert filepath.endswith('.bmp')
    with open(filepath, 'rb') as fid:
        z0 = fid.read()[:30]
    info = struct.unpack('<ccIIIIIIHH', z0)
    ret = {
        'type': info[0]+info[1], #"BM" for windows bitmap, "BA" for OS/2 bitmap
        'size': info[2],
        'reserved_key0': info[3],
        'shift': info[4],
        'len_header': info[5],
        'fig_width': info[6],
        'fig_height': info[7],
        'reserved_key1': info[8],
        'num_color': info[9],
    }
    return ret


def demo_detect_bmp_info():
    print(detect_bmp_info('demo_struct.bmp'))
