'''link: https://matplotlib.org/gallery/images_contours_and_fields/image_demo.html'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


def demo_fxy_function():
    x,y = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    z = np.exp(-x**2-y**2) - np.exp(-(x-1)**2-(y-1)**2)
    fig,ax = plt.subplots()
    ax.imshow(z, interpolation='bilinear', cmap=plt.get_cmap('RdYlGn'), origin='lower', extent=[-3,3,-3,3])


def demo_interpolation():
    z = np.random.rand(5,5)
    fig,(ax0,ax1,ax2) = plt.subplots(1, 3, figsize=(10,3))
    ax0.imshow(z, interpolation='nearest')
    ax0.set_title('nearest')
    ax0.grid()
    ax1.imshow(z, interpolation='bilinear')
    ax1.set_title('bilinear')
    ax1.grid()
    ax2.imshow(z, interpolation='bicubic')
    ax2.set_title('bicubic')
    ax2.grid()
