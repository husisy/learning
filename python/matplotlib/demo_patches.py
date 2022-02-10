import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def demo_patches_ellipse():
    fig,ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    for _ in range(250):
        elli = matplotlib.patches.Ellipse(xy=np.random.rand(2)*10, width=np.random.rand(),
                        height=np.random.rand(), angle=np.random.rand()*360)
        elli.set_clip_box(ax.bbox)
        elli.set_alpha(np.random.rand())
        elli.set_facecolor(np.random.rand(3))
        ax.add_artist(elli)
    ax.set_aspect('equal')
