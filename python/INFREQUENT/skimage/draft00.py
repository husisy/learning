import numpy as np
from skimage.transform import resize

hfe = lambda x,y,eps=1e-3:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def ski_resize(N1=3, N2=5, N3=7, N4=11):
    np1 = np.random.randint(0, 255, size=(N1,N2))

    ind1_x = (np.arange(-1, N1+1)*2+1)/2/N1
    ind2_y = (np.arange(-1, N2+1)*2+1)/2/N2
    np2 = np.pad(np1, [[1,1],[1,1]], mode='symmetric')

    ind3_x = (np.arange(0, N3)*2+1)/2/N3
    ind4_y = (np.arange(0, N4)*2+1)/2/N4
    ret = np.zeros((N3,N4))

    ind3_x_ind1 = np.array([np.where(ind1_x<x)[0][-1] for x in ind3_x])
    ind3_x1 = ind1_x[ind3_x_ind1]
    ind3_x2 = ind1_x[ind3_x_ind1+1]

    ind4_y_ind2 = np.array([np.where(ind2_y<y)[0][-1] for y in ind4_y])
    ind4_y1 = ind2_y[ind4_y_ind2]
    ind4_y2 = ind2_y[ind4_y_ind2+1]

    # reference: https://en.wikipedia.org/wiki/Bilinear_interpolation
    tmp1 = (ind3_x2 - ind3_x)[:,np.newaxis]
    tmp2 = (ind3_x - ind3_x1)[:,np.newaxis]
    tmp3 = (ind4_y2 - ind4_y)[np.newaxis]
    tmp4 = (ind4_y - ind4_y1)[np.newaxis]
    tmp5 = np2[ind3_x_ind1[:,np.newaxis], ind4_y_ind2[np.newaxis]]
    tmp6 = np2[ind3_x_ind1[:,np.newaxis], ind4_y_ind2[np.newaxis]+1]
    tmp7 = np2[ind3_x_ind1[:,np.newaxis]+1, ind4_y_ind2[np.newaxis]]
    tmp8 = np2[ind3_x_ind1[:,np.newaxis]+1, ind4_y_ind2[np.newaxis]+1]
    tmp9 = ((ind3_x2-ind3_x1)[:,np.newaxis]) * ((ind4_y2-ind4_y1)[np.newaxis])
    np3 = ((tmp1*tmp5+tmp2*tmp7)*tmp3 + (tmp1*tmp6+tmp2*tmp8)*tmp4)/tmp9

    ski1 = resize(np1.astype(np.uint8), [N3,N4], mode='symmetric', anti_aliasing=False)*255
    print('resize image:: np vs skimage: ', hfe(np3, ski1))


def ski_misc00():
    np1 = np.random.randint(0,255,size=(2,2),dtype=np.uint8)
    np2 = np.random.randint(0,255,size=(3,3),dtype=np.uint8)
    np2[1:3,1:3] = np1
    np1_ = resize(np1, (4,4), mode='symmetric', anti_aliasing=False)
    np2_ = resize(np2, (6,6), mode='symmetric', anti_aliasing=False)
    np3 = np.zeros(np2_.shape)
    np3[2:6,2:6] = np1_
    z1 = np3-np2_
    z1[np.abs(z1)<1e-7] = 0
