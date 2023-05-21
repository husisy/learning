import sys
import math
import struct
import random
import base64

def uint32_to_bytes(x):
    # big-endian
    assert 0 <= x < 2**32
    ret = bytes([
        (x & 0xff000000) >> 24,
        (x & 0xff0000) >> 16,
        (x & 0xff00) >> 8,
        (x & 0xff),
    ])
    return ret


def random_bytes(length=1, seed=None):
    if seed is not None:
        state = random.getstate()
        random.seed(seed)
    assert length>=0
    ret = bytes(random.randint(0,255) for _ in range(length))
    if seed is not None:
        random.setstate(state)
    return ret


def show_bytes(x, desc='', tag_detail=False):
    # byte: raw, int (big-endian), int (little-endian), uint8 list, binary str, hex str, base64 byte
    assert isinstance(x, bytes)
    x_int_big_endian = int.from_bytes(x, 'big')
    x_int_little_endian = int.from_bytes(x, 'little')
    x_uint8_list = list(x)
    x_bin = bin(x_int_big_endian)[2:].rjust(8*len(x), '0')
    x_hex = x.hex()
    x_base64 = base64.urlsafe_b64encode(x)
    assert base64.urlsafe_b64decode(x_base64)==x
    space_prefix = ' '*(len(desc)+4)
    print(f'{desc}raw: {x}')
    print(f'{desc}binary str: {x_bin}')
    print(f'{desc}uint8 list: {x_uint8_list}')
    if tag_detail:
        tmp0 = ' '.join([f'{x_bin[(8*y0):(8*y0+8)]}={y1}' for y0,y1 in enumerate(x_uint8_list)])
        print(f'{space_prefix} {tmp0}')
    print(f'{desc}hex str: {x_hex}')
    if tag_detail:
        tmp0 = ' '.join([f'{x_bin[(4*y0):(4*y0+4)]}={y1}' for y0,y1 in enumerate(x_hex)])
        print(f'{space_prefix} {tmp0}')
    print(f'{desc}int (big-endian): {x_int_big_endian}')
    if tag_detail:
        tmp0 = ' + '.join([f'{y1}*256**{len(x)-y0-1}' for y0,y1 in enumerate(x_uint8_list)])
        print(f'{space_prefix} {tmp0} = {x_int_big_endian}')
    print(f'{desc}int (little-endian): {x_int_little_endian}')
    if tag_detail:
        tmp0 = ' + '.join([f'{y1}*256**{y0}' for y0,y1 in enumerate(x_uint8_list)])
        print(f'{space_prefix} {tmp0} = {x_int_little_endian}')
    print(f'{desc}base64 byte: {x_base64}')


def demo_show_bytes():
    x_byte = random_bytes(3, seed=234)
    show_bytes(x_byte, tag_detail=True)
    # raw: b'\xaf\x87\x1d'
    # binary str: 101011111000011100011101
    # uint8 list: [175, 135, 29]
    #     10101111=175 10000111=135 00011101=29
    # hex str: af871d
    #     1010=a 1111=f 1000=8 0111=7 0001=1 1101=d
    # int (big-endian): 11503389
    #     175*256**2 + 135*256**1 + 29*256**0 = 11503389
    # int (little-endian): 1935279
    #     175*256**0 + 135*256**1 + 29*256**2 = 1935279
    # base64 byte: b'r4cd'

def demo_str_bytes():
    # ascii charactors: str, byte,
    x_str = 'hello'
    x_byte = x_str.encode('utf-8')
    assert x_str==x_byte.decode('utf-8')
    print('ascii str:', x_str)
    print('utf-8 encoded byte:', x_byte)
    show_bytes(x_byte, desc='    ')
    # ascii str: hello
    # utf-8 encoded byte: b'hello'
    #      raw: b'hello'
    #      binary str: 0110100001100101011011000110110001101111
    #      uint8 list: [104, 101, 108, 108, 111]
    #      hex str: 68656c6c6f
    #      int (big-endian): 448378203247
    #      int (little-endian): 478560413032
    #      base64 byte: b'aGVsbG8='

    x_str = '你好'
    x_byte = x_str.encode('utf-8')
    assert x_str==x_byte.decode('utf-8')
    print('ascii str:', x_str)
    print('utf-8 encoded byte:', x_byte)
    show_bytes(x_byte, desc='    ')
    # ascii str: 你好
    # utf-8 encoded byte: b'\xe4\xbd\xa0\xe5\xa5\xbd'
    #     raw: b'\xe4\xbd\xa0\xe5\xa5\xbd'
    #     binary str: 111001001011110110100000111001011010010110111101
    #     uint8 list: [228, 189, 160, 229, 165, 189]
    #     hex str: e4bda0e5a5bd
    #     int (big-endian): 251503099356605
    #     int (little-endian): 208520219770340
    #     base64 byte: b'5L2g5aW9'


def demo_int_bytes():
    sys.byteorder # little big
    x_int = random.randint(2**24, 2**32-1) #make sure 4 bytes

    x_byte_big_endian = x_int.to_bytes(4, byteorder='big')
    assert x_int==int.from_bytes(x_byte_big_endian, byteorder='big')
    print('int:', x_int)
    print('big endian byte:', x_byte_big_endian)
    show_bytes(x_byte_big_endian, desc='    ', tag_detail=True)
    # int: 3779619461
    # big endian byte: b'\xe1Hj\x85'
    #     raw: b'\xe1Hj\x85'
    #     binary str: 11100001010010000110101010000101
    #     uint8 list: [225, 72, 106, 133]
    #         11100001=225 01001000=72 01101010=106 10000101=133
    #     hex str: e1486a85
    #         1110=e 0001=1 0100=4 1000=8 0110=6 1010=a 1000=8 0101=5
    #     int (big-endian): 3779619461
    #         225*256**3 + 72*256**2 + 106*256**1 + 133*256**0 = 3779619461
    #     int (little-endian): 2238335201
    #         225*256**0 + 72*256**1 + 106*256**2 + 133*256**3 = 2238335201
    #     base64 byte: b'4UhqhQ=='

    x_byte_little_endian = x_int.to_bytes(4, byteorder='little')
    assert x_int==int.from_bytes(x_byte_little_endian, byteorder='little')
    print('int:', x_int)
    print('little endian byte:', x_byte_little_endian)
    show_bytes(x_byte_little_endian, desc='    ', tag_detail=True)
    # int: 3779619461
    # little endian byte: b'\x85jH\xe1'
    #     raw: b'\x85jH\xe1'
    #     binary str: 10000101011010100100100011100001
    #     uint8 list: [133, 106, 72, 225]
    #         10000101=133 01101010=106 01001000=72 11100001=225
    #     hex str: 856a48e1
    #         1000=8 0101=5 0110=6 1010=a 0100=4 1000=8 1110=e 0001=1
    #     int (big-endian): 2238335201
    #         133*256**3 + 106*256**2 + 72*256**1 + 225*256**0 = 2238335201
    #     int (little-endian): 3779619461
    #         133*256**0 + 106*256**1 + 72*256**2 + 225*256**3 = 3779619461
    #     base64 byte: b'hWpI4Q=='

    # negative integer
    x = -3
    x_byte = x.to_bytes(1, byteorder='big', signed=True)
    assert x==(x_byte[0]-256)


def demo_float():
    # endianess matter
    x_double = random.random()
    x_bin = struct.pack('<d', x_double)
    # TODO https://en.wikipedia.org/wiki/Double-precision_floating-point_format


def demo_pack_unpack():
    # >: big-endian
    # <: little-endian
    # I: 4-byte unsigned integer
    # H: 2-byte unsigned integer
    x = random.randint(0, 2**32-1)
    tmp0 = x.to_bytes(4, byteorder='big')
    tmp1 = struct.pack('>I', x)
    assert tmp0==tmp1
    assert struct.unpack('>I', tmp0)[0]==x

    x0 = random.randint(0, 2**32-1)
    x1 = random.randint(0, 2**16-1)
    tmp0 = x0.to_bytes(4, byteorder='big') + x1.to_bytes(2, byteorder='big')
    tmp1 = struct.pack('>IH', x0, x1)
    assert tmp0==tmp1
    assert struct.unpack('>IH', tmp0)==(x0,x1)


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
