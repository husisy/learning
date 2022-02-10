import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

# TODO axvline axhline
# TODO axhspan axvspan
# TODO demo_xxx
# TODO set_xticks set_xticklabels

def demo_basic00():
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(x, y, label='line0')
    ax.plot(x, y+1, label='line1')
    # useful keywords: linestyle linewidth markerstyle markersize markevery
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.set_title('this is axes title')
    ax.legend()
    fig.suptitle('this is figure title')

    # close windows is not enough, TODO add link
    plt.close(fig)
    # plt.close(plt.gcf())
    # plt.close('all')


def demo_basic_plot_type():
    # hist
    fig,ax = plt.subplots()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.hist(np.random.randn(500), bins=50, density=True)
    ax.set_title('.hist()')
    plt.show()

    # bar
    N0 = 5
    mean1 = np.random.randint(50, 100, [N0])
    std1 = np.random.randint(5, 10, [N0])
    mean2 = np.random.randint(60, 80, [N0])
    std2 = np.random.randint(2, 5, [N0])
    fig,ax = plt.subplots()
    bar_width = 0.35
    tmp0 = dict(width=bar_width, alpha=0.4, error_kw={'ecolor':'0.3'})
    ax.bar(np.arange(N0), mean1, color='b', yerr=std1, label='bar0', **tmp0)
    ax.bar(np.arange(N0)+bar_width, mean2, color='r', yerr=std2, label='bar1', **tmp0)
    ax.set_xticks(np.arange(N0) + bar_width/2)
    ax.set_xticklabels(['x'+str(ind1) for ind1 in range(N0)])
    ax.legend()
    ax.set_title('.bar()')


def demo_plot_manage_spline():
    x = np.linspace(-np.pi, np.pi, 300)
    y1 = np.cos(x)
    y2 = np.sin(x)

    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(x, y1, linewidth=1, label='cosine')
    ax.plot(x, y2, linewidth=1, label='sine')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([r'$-1$', r'$0$', r'$+1$'])
    ax.legend(loc='upper left', frameon=False)

    # moving spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    # plt.annotate
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))


def demo_axis_formatter():
    fig,ax = plt.subplots()
    ax.plot([0,1], [0,1])
    hf0 = lambda x,pos: 'a{:1.1f}bb'.format(x)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(hf0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('b%1.2fcc'))


def demo_streamplot():
    x,y = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    u = -1 - x**2 + y
    v = 1 + x - y**2
    speed = np.sqrt(u**2 + v**2)

    fig = plt.figure()
    (ax0,ax1),(ax2,ax3) = fig.subplots(2,2)

    ax0.streamplot(x, y, u, v, density=[0.5,1])
    ax0.set_title('varying density')

    strm = ax1.streamplot(x, y, u, v, color=u, linewidth=2, cmap='autumn')
    fig.colorbar(strm.lines, ax=ax1)
    ax1.set_title('varying color')

    ax2.streamplot(x, y, u, v, density=0.6, color='k', linewidth=5*speed/speed.max())
    ax2.set_title('varying line width')

    seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])

    strm = ax3.streamplot(x, y, u, v, color=u, linewidth=2, cmap='autumn', start_points=seed_points.T)
    fig.colorbar(strm.lines, ax=ax3)
    ax3.set_title('controlling starting points')


def demo_style():
    x = np.linspace(0, 2*np.pi)
    y = np.sin(x)
    with plt.style.context('dark_background'):
        fig,ax = plt.subplots()
        ax.plot(x, y, 'r-o')


def demo_contourf():
    # see https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html
    x,y = np.meshgrid(np.linspace(1,5,100), np.linspace(1,5,90))
    z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
    fig,ax = plt.subplots()
    level = matplotlib.ticker.MaxNLocator(nbins=15).tick_values(z.min(), z.max())
    hContourf = ax.contourf(x, y, z, levels=level, cmap=plt.get_cmap('PiYG'))
    fig.colorbar(hContourf, ax=ax)


def demo_colorbar():
    fig,ax = plt.subplots()
    image = ax.imshow(np.arange(100).reshape((10, 10)))
    fig.colorbar(image)


def demo_colorbar_multiple_image():
    fig, tmp0 = plt.subplots(2, 2)
    ax_list = [tmp0[0,0], tmp0[0,1], tmp0[1,0], tmp0[1,1]]
    image_list = [ax.imshow(np.random.uniform(x,x+1, size=(10,20)), cmap='cool') for x,ax in enumerate(ax_list)]
    tmp0 = min(x.get_array().min() for x in image_list)
    tmp1 = max(x.get_array().max() for x in image_list)
    norm = matplotlib.colors.Normalize(vmin=tmp0, vmax=tmp1)
    for x in image_list:
        x.set_norm(norm)
    fig.colorbar(image_list[0], ax=ax_list, orientation='horizontal', fraction=0.1)
    for x in ax_list:
        x.xaxis.set_visible(False)
        x.yaxis.set_visible(False)


def demo_xkcd():
    xdata = np.linspace(0, 2*np.pi, 50)
    ydata = np.sin(5*xdata) - 0.5*xdata
    with plt.xkcd():
        fig,ax = plt.subplots()
        ax.plot(xdata, ydata)
        ax.set_xlabel('time')
        ax.set_title('personal emotion state')
        fig.tight_layout()


def demo_multicolored_line():
    xdata = np.linspace(0, 3*np.pi, 500)
    ydata = np.sin(xdata)
    cdata = np.cos((xdata[1:]+xdata[:-1])/2) #use first derivative as color

    fig,ax = plt.subplots()
    tmp0 = np.stack([xdata,ydata],axis=1)
    segments = np.stack([tmp0[:-1],tmp0[1:]], axis=1)
    norm = plt.Normalize(cdata.min(), cdata.max())
    lc = matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(cdata)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    ax.set_xlim(xdata.min(), xdata.max())
    ax.set_ylim(ydata.min()-0.1, ydata.max()+0.1)
