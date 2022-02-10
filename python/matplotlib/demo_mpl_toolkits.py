import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits
import mpl_toolkits.axes_grid1
import mpl_toolkits.axes_grid1.inset_locator
plt.ion()


def demo_inset_locator():
    xdata = np.linspace(-np.pi, np.pi, 500)
    ydata = np.sin(xdata*30) * xdata / 100
    fig = plt.figure()

    ax0 = fig.subplots(1,1)
    ax0.plot(xdata, ydata)
    ax0.grid(True)

    ax1 = mpl_toolkits.axes_grid1.inset_locator.zoomed_inset_axes(ax0, 3, loc='upper right')
    ax1.plot(xdata, ydata)
    ax1.grid(True)
    ax1.set_xlim(-0.3, 0.4)
    ax1.set_ylim(-0.005, 0.005)
    ax1.yaxis.get_major_locator().set_params(nbins=7)
    ax1.xaxis.get_major_locator().set_params(nbins=7)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    mpl_toolkits.axes_grid1.inset_locator.mark_inset(ax0, ax1, loc1=2, loc2=4, fc='none', ec='0.5')


def demo_locatable_colorbar():
    fig,ax = plt.subplots()
    image = ax.imshow(np.arange(100).reshape((10, 10)))
    cax = mpl_toolkits.axes_grid1.make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(image, cax=cax)
