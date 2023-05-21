import sys
import base64
import numpy as np
# see python/STDL/draft_struct.py

np_i_type = {np.int8,np.int16,np.int32,np.int64}
np_f_type = {np.float16,np.float32,np.float64}
np_ui_type = {np.uint8,np.uint16,np.uint32,np.uint64}
np_i_to_ui = {np.int8:np.uint8, np.int16:np.uint16, np.int32:np.uint32, np.int64:np.uint64}
np_f_to_ui = {np.float16:np.uint16, np.float32:np.uint32, np.float64:np.uint64}

# https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
endianness_map = {
    '>': 'big',
    '<': 'little',
    '=': sys.byteorder,
    '|': 'not applicable',
}


# from python/STDL/draft_struct.py
def show_bytes(x, itemsize, desc='', tag_detail=False):
    # byte: raw, int (big-endian), int (little-endian), uint8 list, binary str, hex str, base64 byte
    assert isinstance(x, bytes)
    assert len(x)%itemsize==0
    x_list = [x[y:(y+itemsize)] for y in range(0,len(x),itemsize)]
    for ind0,x in enumerate(x_list):
        x_int_big_endian = int.from_bytes(x, 'big')
        x_int_little_endian = int.from_bytes(x, 'little')
        x_uint8_list = list(x)
        x_bin = bin(x_int_big_endian)[2:].rjust(8*len(x), '0')
        x_hex = x.hex()
        # x_base64 = base64.urlsafe_b64encode(x)
        # assert base64.urlsafe_b64decode(x_base64)==x
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
        tmp0 = 'int (big-endian)' if itemsize>1 else 'int'
        print(f'{desc}{tmp0}: {x_int_big_endian}')
        if tag_detail:
            tmp0 = ' + '.join([f'{y1}*256**{len(x)-y0-1}' for y0,y1 in enumerate(x_uint8_list)])
            print(f'{space_prefix} {tmp0} = {x_int_big_endian}')
        if itemsize>1:
            print(f'{desc}int (little-endian): {x_int_little_endian}')
            if tag_detail:
                tmp0 = ' + '.join([f'{y1}*256**{y0}' for y0,y1 in enumerate(x_uint8_list)])
                print(f'{space_prefix} {tmp0} = {x_int_little_endian}')
        # print(f'{desc}base64 byte: {x_base64}')
        if ind0!=len(x_list)-1:
            tmp0 = '-'*10
            print(f'{desc}{tmp0}')


def hfb(np0):
    # big-endian
    np0 = np.asarray(np0)
    ndim = np0.ndim
    assert ndim in {0,1}
    nptype = np0.dtype.type
    if nptype in np_i_to_ui:
        np0 = np0.view(np_i_to_ui[nptype])
    if nptype in np_f_to_ui:
        np0 = np0.view(np_f_to_ui[nptype])
    nptype = np0.dtype.type
    assert nptype in np_ui_type
    ret = [np.binary_repr(x, width=np0.itemsize*8) for x in np0.reshape(-1)]
    if ndim==0:
        ret = ret[0]
    return ret


def test_tobytes_from_buffer():
    np_rng = np.random.default_rng()
    N0 = 23
    dtype_int_list = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
    for dtype in dtype_int_list:
        np0 = np_rng.integers(0, 128, size=N0, dtype=dtype)
        np0_byte = np0.tobytes()
        np1 = np.frombuffer(np0_byte, dtype=dtype)
        assert np.array_equal(np0, np1)
    dtype_float_list = [np.float16, np.float32, np.float64]
    for dtype in dtype_float_list:
        np0 = np_rng.uniform(0, 1, size=N0).astype(dtype)
        np0_byte = np0.tobytes()
        np1 = np.frombuffer(np0_byte, dtype=dtype)
        assert np.array_equal(np0, np1)


def demo_uint8_array():
    np_rng = np.random.default_rng(233)
    np0 = np_rng.integers(0, 256, size=(2,2), dtype=np.uint8)
    print('array:', np0)
    print('endianess:', endianness_map[np0.dtype.byteorder])
    # array: [[174 132]
    # [ 99  25]]
    # endianess: not applicable

    np0_byte_c = np0.tobytes(order='C') #default
    print('byte order=c:', np0_byte_c)
    show_bytes(np0_byte_c, np0.itemsize, desc='    ')
    # byte order=c: b'\xae\x84c\x19'
    #     raw: b'\xae'
    #     binary str: 10101110
    #     uint8 list: [174]
    #     hex str: ae
    #     int: 174
    #     ----------
    #     raw: b'\x84'
    #     binary str: 10000100
    #     uint8 list: [132]
    #     hex str: 84
    #     int: 132
    #     ----------
    #     raw: b'c'
    #     binary str: 01100011
    #     uint8 list: [99]
    #     hex str: 63
    #     int: 99
    #     ----------
    #     raw: b'\x19'
    #     binary str: 00011001
    #     uint8 list: [25]
    #     hex str: 19
    #     int: 25

    np0_byte_f = np0.tobytes(order='F')
    print('array:', np0)
    print('byte order=F:', np0_byte_f)
    show_bytes(np0_byte_f, np0.itemsize, desc='    ')
    # byte order=F: b'\xaec\x84\x19'
    #     raw: b'\xae'
    #     binary str: 10101110
    #     uint8 list: [174]
    #     hex str: ae
    #     int: 174
    #     ----------
    #     raw: b'c'
    #     binary str: 01100011
    #     uint8 list: [99]
    #     hex str: 63
    #     int: 99
    #     ----------
    #     raw: b'\x84'
    #     binary str: 10000100
    #     uint8 list: [132]
    #     hex str: 84
    #     int: 132
    #     ----------
    #     raw: b'\x19'
    #     binary str: 00011001
    #     uint8 list: [25]
    #     hex str: 19
    #     int: 25


def demo_uint32_byte():
    np_rng = np.random.default_rng(233)
    np0 = np_rng.integers(0, 2**32, size=3, dtype=np.uint32)
    np0_byte = np0.tobytes() #order='C' is the same as the order='F' for 1d array
    print('array:', np0)
    print('endianess:', endianness_map[np0.dtype.byteorder])
    print('byte:', np0_byte)
    show_bytes(np0_byte, np0.itemsize, desc='    ', tag_detail=True)
    # array: [425952430 405934077 329590467]
    # endianess: little
    # byte: b'\xae\x84c\x19\xfd\x0f2\x18\xc3&\xa5\x13'
    #     raw: b'\xae\x84c\x19'
    #     binary str: 10101110100001000110001100011001
    #     uint8 list: [174, 132, 99, 25]
    #         10101110=174 10000100=132 01100011=99 00011001=25
    #     hex str: ae846319
    #         1010=a 1110=e 1000=8 0100=4 0110=6 0011=3 0001=1 1001=9
    #     int (big-endian): 2927911705
    #         174*256**3 + 132*256**2 + 99*256**1 + 25*256**0 = 2927911705
    #     int (little-endian): 425952430
    #         174*256**0 + 132*256**1 + 99*256**2 + 25*256**3 = 425952430
    #     ----------
    #     raw: b'\xfd\x0f2\x18'
    #     binary str: 11111101000011110011001000011000
    #     uint8 list: [253, 15, 50, 24]
    #         11111101=253 00001111=15 00110010=50 00011000=24
    #     hex str: fd0f3218
    #         1111=f 1101=d 0000=0 1111=f 0011=3 0010=2 0001=1 1000=8
    #     int (big-endian): 4245631512
    #         253*256**3 + 15*256**2 + 50*256**1 + 24*256**0 = 4245631512
    #     int (little-endian): 405934077
    #         253*256**0 + 15*256**1 + 50*256**2 + 24*256**3 = 405934077
    #     ----------
    #     raw: b'\xc3&\xa5\x13'
    #     binary str: 11000011001001101010010100010011
    #     uint8 list: [195, 38, 165, 19]
    #         11000011=195 00100110=38 10100101=165 00010011=19
    #     hex str: c326a513
    #         1100=c 0011=3 0010=2 0110=6 1010=a 0101=5 0001=1 0011=3
    #     int (big-endian): 3274089747
    #         195*256**3 + 38*256**2 + 165*256**1 + 19*256**0 = 3274089747
    #     int (little-endian): 329590467
    #         195*256**0 + 38*256**1 + 165*256**2 + 19*256**3 = 329590467


def demo_view_byteorder():
    np0 = np.array([0, 3, 7, 15], dtype=np.uint8)
    np1 = np0.view(np.uint16) #doesn't change the byte order
    for x in [np0,np1]:
        x_byte = x.tobytes()
        print('array:', x)
        print('endianess:', endianness_map[x.dtype.byteorder])
        print('byte:', x_byte)
        show_bytes(x_byte, x.itemsize, desc='    ')
    # array: [ 0  3  7 15]
    # endianess: not applicable
    # byte: b'\x00\x03\x07\x0f'
    #     raw: b'\x00'
    #     binary str: 00000000
    #     uint8 list: [0]
    #     hex str: 00
    #     int: 0
    #     ----------
    #     raw: b'\x03'
    #     binary str: 00000011
    #     uint8 list: [3]
    #     hex str: 03
    #     int: 3
    #     ----------
    #     raw: b'\x07'
    #     binary str: 00000111
    #     uint8 list: [7]
    #     hex str: 07
    #     int: 7
    #     ----------
    #     raw: b'\x0f'
    #     binary str: 00001111
    #     uint8 list: [15]
    #     hex str: 0f
    #     int: 15
    # array: [ 768 3847]
    # endianess: little
    # byte: b'\x00\x03\x07\x0f'
    #     raw: b'\x00\x03'
    #     binary str: 0000000000000011
    #     uint8 list: [0, 3]
    #     hex str: 0003
    #     int (big-endian): 3
    #     int (little-endian): 768
    #     ----------
    #     raw: b'\x07\x0f'
    #     binary str: 0000011100001111
    #     uint8 list: [7, 15]
    #     hex str: 070f
    #     int (big-endian): 1807
    #     int (little-endian): 3847


def demo_float16():
    np0 = np.zeros(1, dtype=np.float16)
    np1 = np.nextafter(np0, np0+1)
    np2 = np.nextafter(np0, np0-1)
    print('float16(+delta)', np1.item(), hfb(np1))
    print('float16(-delta)', np2.item(), hfb(np2))

    np0 = np.ones(1, dtype=np.float16)
    np1 = np.nextafter(np0, np0+1)
    np2 = np.nextafter(np0, np0-1)
    print('float16(1+delta)', np1.item(), hfb(np1))
    print('float16(1-delta)', np2.item(), hfb(np2))



def test_unpackbits():
    np_rng = np.random.default_rng()
    N0 = 3
    np0 = np_rng.integers(0, 256, size=N0, dtype=np.uint8)
    for bitorder in ['big', 'little']:
        np1 = np.unpackbits(np0, axis=0, bitorder=bitorder)
        assert np1.shape==(N0*8,)
        tmp0 = [bin(x)[2:].rjust(8, '0') for x in np0.tobytes()]
        if bitorder=='little':
            tmp0 = [x[::-1] for x in tmp0]
        np2 = np.array([y=='1' for x in tmp0 for y in x], dtype=np.uint8)
        assert np.array_equal(np1, np2)

        tmp0 = np.packbits(np1, axis=0, bitorder=bitorder)
        assert np.array_equal(tmp0, np0)
