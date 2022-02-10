import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def demo_dirichlet_distribution():
    # https://en.wikipedia.org/wiki/Dirichlet_distribution
    np_rng = np.random.default_rng()
    alpha_list = [
        (1, 1, 1),
        (10, 10, 10),
        (0.1, 0.1, 0.1),
        (1, 1, 10),
    ]
    N0 = int(1e4)

    fig,tmp0 = plt.subplots(2, 2)
    ax_list = [tmp0[0][0],tmp0[0][1],tmp0[1][0],tmp0[1][1]]
    for alpha,ax in zip(alpha_list,ax_list):
        xdata = np_rng.dirichlet(alpha, size=N0)
        ax.scatter(xdata[:,0]+0.5*xdata[:,1], xdata[:,1], s=2, alpha=0.8, linewidth=0, edgecolors='none')
        ax.plot([0,0.5], [0,1], color='k')
        ax.plot([1,0.5], [0,1], color='k')
        ax.plot([0, 1], [0,0], color='k')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.001, 1)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_title(str(alpha))
    fig.tight_layout()
