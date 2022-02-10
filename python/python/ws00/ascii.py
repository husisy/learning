import argparse
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np


def gray2ascii(img, ascii_char):
    '''
    img_gray(float, (N1,N2), [0,1))

    ascii_char(list/(N3))
    '''
    N1, N2 = img.shape[0:2]
    img_alpha = img[:,:,3] if img.shape[2]==4 else np.ones((N1,N2), dtype=np.uint8)
    N3 = len(ascii_char)
    img_gray = np.floor(rgb2gray(img)*(N3-1)).astype(np.int32)
    txt = ''
    for ind1 in range(N1):
        for ind2 in range(N2):
            txt += ' ' if img_alpha[ind1,ind2]==0 else ascii_char[img_gray[ind1,ind2]]
        txt += '\n'
    return txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-o', '--output')
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--height', type=int, default=80)
    args = parser.parse_args()
    img_file, output_file, height, width = args.file, args.output, args.height, args.width

    img = resize(imread(img_file), (height, width))
    tmp1 = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
    txt = gray2ascii(img, tmp1)
    print(txt)

    if output_file:
        with open(output_file, 'w') as fid:
            fid.write(txt)
    else:
        with open("output.txt", 'w') as fid:
            fid.write(txt)
