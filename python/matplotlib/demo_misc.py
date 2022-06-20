import io
import PIL.Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def demo_misc00():
    print(plt.style.available)
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
    # plt.style.use('fivethirtyeight')

    fig = plt.figure()
    print(fig.canvas.get_supported_filetypes())

    print(matplotlib.rcParams['lines.linewidth'])
    print(matplotlib.rcParams['lines.color'])


def plt_fig_to_nparray(fig, package='pillow'):
    '''
    pillow solution see: https://stackoverflow.com/a/32908899
    tensorflow solution see: https://www.tensorflow.org/tensorboard/r2/image_summaries#logging_arbitrary_image_data
    '''
    import io
    assert package in {'pillow','tensorflow'} # both methods give same results
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    if package=='pillow':
        from PIL import Image
        with Image.open(buffer) as image_pil:
            ret = np.array(image_pil)
    else:
        import tensorflow as tf
        ret = tf.image.decode_png(buffer.getvalue(), channels=4).numpy()
    return ret

def demo_plt_fig_to_nparray():
    np0 = np.linspace(0, np.pi*4, 1000)
    fig, ax = plt.subplots(1, 1)
    ax.plot(np0, np.sin(np0))

    np1 = plt_fig_to_nparray(fig, package='pillow') #np.uint8
    np2 = plt_fig_to_nparray(fig, package='tensorflow') #np.uint8
    plt.close(fig)
    assert hfe(np1.astype(np.float64), np2.astype(np.float64)) < 1e-10
    fig,ax = plt.subplots()
    ax.imshow(np1)


def demo_plot_matrix():
    def plot_rectangle(ax, x, y, deltax=1, deltay=1, **kwargs):
        tmp0 = dict(facecolor=[1,1,1], alpha=1, linewidth=1, edgecolor='black')
        tmp0.update(kwargs)
        rect = plt.Rectangle((x, y), deltax, deltay, **tmp0)
        ax.add_patch(rect)

    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]

    num_row = 4
    num_column = 8

    fig,ax = plt.subplots()
    plot_rectangle(ax, 1, 1, num_column, num_row)
    plot_rectangle(ax, 0.5, 0.5, num_column, num_row)
    for x in range(num_column):
        plot_rectangle(ax, x, num_row-1, facecolor=tableau_colorblind[7])
    for y in range(num_row-1):
        for x in range(num_column):
            plot_rectangle(ax, x, y)

    ax.set_xlim(-1, num_column+1)
    ax.set_ylim(-1, num_row+1)
    ax.set_aspect('equal')
    ax.axis('off')


def demo_center_spines_with_arrow():
    # https://matplotlib.org/stable/gallery/ticks_and_spines/centered_spines_with_arrows.html
    xdata = np.linspace(-10, 10, 100)
    ydata = 1/(1+np.exp(-xdata)) - 0.5

    fig,ax = plt.subplots(1,1)
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.plot(xdata, ydata)
    ax.set_ylim(-0.7, 0.7)


def plt_txt_to_nparray(text, w_pixel, figsize=(3,2), fontsize=48):
    # plt.rcParams['font.sans-serif'] = ['SimHei'] for Chinese character
    fig,ax = plt.subplots(figsize=figsize)
    ax.text(0, 0, text, fontsize=fontsize, verticalalignment='center', horizontalalignment='center', weight='bold')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_pil = PIL.Image.open(buffer).convert('L')
    h1 = int(image_pil.height*w_pixel/image_pil.width)
    ret = np.array(image_pil.resize((w_pixel,h1)))
    return ret

def demo_plt_txt_to_nparray():
    plt_txt_to_nparray('hello\nworld', 480)
