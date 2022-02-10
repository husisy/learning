import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageENhance

hf_file = lambda *x: os.path.join('tbd00', *x)
hf_data = lambda *x: os.path.join('data', *x)

def random_image_filepath(_dir='data'):
    tmp0 = [os.path.join(_dir,x) for x in os.listdir(_dir) if x.endswith('.jpg')]
    ret = random.choice(tmp0)
    return ret

z0 = Image.open(random_image_filepath()) #only read the file header at this step
z0.format #JPEG PPM None(when not read from a file)
z0.size #(1920,1200) (width,height)
z0.mode #RGB L(luminance) CMYK
# z0.show() #will use system software to open it (windows-Photos)
# z0.save(hf_file('tbd00.png')) #thefilename extension is important

## image processing
z0.thumbnail((128,128))
z1 = z0.crop(box=(100,100,400,400)) #left upper right lower
# z0.paste(z1, box=(100,100,400,400))
z1 = z0.transpose(Image.ROTATE_180)
# z0.resize
# z0.rotate

r,g,b = z0.split()
z1 = Image.merge('RGB', (r,g,b)) #(PPM, other) <-> (RGB,L)
z0.convert('L')

## image enhancement
z1 = z0.filter(ImageFilter.DETAIL)
# z1 = z0.point(lambda x: x*1.2)

## image sequence

## numpy
hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
np0 = np.array(z0) #(np,uint8,(1920,1200,3))
np1 = np.array(Image.fromarray(np0))
hfe(np0.astype(np.float), np1.astype(np.float))
