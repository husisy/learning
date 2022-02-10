import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y,),5)


def plt_fig_to_nparray(fig, package='pillow'):
    '''
    pillow solution see: https://stackoverflow.com/a/32908899
    tensorflow solution see: https://www.tensorflow.org/tensorboard/r2/image_summaries#logging_arbitrary_image_data
    '''
    assert package in {'pillow','tensorflow'} # both methods give same results
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    if package=='pillow':
        with Image.open(buffer) as image_pil:
            ret = np.array(image_pil)
    else:
        ret = tf.image.decode_png(buffer.getvalue(), channels=4).numpy()
    return ret

def example_plt_fig_to_nparray():
    plt.ioff()
    np1 = np.linspace(0, np.pi*4, 1000)
    np2 = np.sin(np1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(np1, np2)

    np3 = plt_fig_to_nparray(fig, package='pillow')
    np4 = plt_fig_to_nparray(fig, package='tensorflow')
    print('example_plt_fig_to_nparray: tensorflow vs pillow: ', hfe_r5(np3.astype(np.float),np4.astype(np.float)))

    fig1, ax1 = plt.subplots(1, 1)
    ax1.imshow(np3)
    plt.show()

if __name__ == "__main__":
    example_plt_fig_to_nparray()
