'''link: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
plt.ion()


def demo_3d_basic_line():
    N0 = 200
    theta = np.linspace(-4*np.pi, 4*np.pi, N0)
    z = np.linspace(-2, 2, N0)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot(x, y, z)


def demo_3d_basic_scatter():
    N0 = 200
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    xyz = np.random.randn(3, N0)
    ax.scatter(*xyz, c='r', marker='o')
    xyz = np.random.randn(3, N0)
    ax.scatter(*xyz, c='b', marker='^')


def demo_3d_basic_surface():
    x,y = np.meshgrid(np.linspace(-5,5), np.linspace(-5,5))
    z = np.sin(np.sqrt(x**2 + y**2))
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    hSurf = ax.plot_surface(x, y, z, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=True)
    fig.colorbar(hSurf, shrink=0.5, aspect=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plt_subplots_3d(nrols=1, ncols=1, **kwargs):
    assert (nrols>=1) and (ncols>=1)
    fig = plt.figure(**kwargs)
    ax_list = []
    grid = matplotlib.gridspec.GridSpec(nrols, ncols)
    hf0 = lambda x,y: grid[x,y].get_position(fig)
    for ind0 in range(nrols):
        tmp0 = []
        for ind1 in range(ncols):
            ax = mpl_toolkits.mplot3d.Axes3D(fig, rect=hf0(ind0,ind1), auto_add_to_figure=False)
            fig.add_axes(ax)
            tmp0.append(ax)
        ax_list.append(tmp0)
    if (nrols==1) ^ (ncols==1): #xor
        ax_list = [y for x in ax_list for y in x] #flat to one-dimension
    if (nrols==1) and (ncols==1):
        ax_list = ax_list[0][0] #flat to zero-dimension
    return fig, ax_list

def demo_3d_subplots_surface():
    x,y = np.meshgrid(np.linspace(-5,5), np.linspace(-5,5))
    z0 = np.sin(np.sqrt(x**2 + y**2))
    z1 = np.cos(np.sqrt(x**2 + y**2))

    fig,(ax0,ax1) = plt_subplots_3d(1, 2, figsize=(10,5))
    hSurf = ax0.plot_surface(x, y, z0, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=True)
    fig.colorbar(hSurf, ax=ax0, shrink=0.5, aspect=5)
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')

    hSurf = ax1.plot_surface(x, y, z1, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=True)
    fig.colorbar(hSurf, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')


def demo_3d_basic_surface01():
    theta = np.linspace(0, np.pi, 100)[np.newaxis]
    phi = np.linspace(0, 2*np.pi, 100)[:,np.newaxis]

    x = 10 * np.sin(theta) * np.cos(phi)
    y = 10 * np.sin(theta) * np.sin(phi)
    z = 10 * np.cos(theta) * np.ones_like(phi)

    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(x, y, z)


def demo_3d_wireframe():
    x, y, z = mpl_toolkits.mplot3d.axes3d.get_test_data(0.05)
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10)


def demo_contour():
    x, y, z = mpl_toolkits.mplot3d.axes3d.get_test_data(0.05)

    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.contour(x, y, z, cmap=plt.get_cmap('coolwarm'))


def demo_plot_trisurf():
    r = np.linspace(0, 1, 10)[:,np.newaxis]
    theta = np.linspace(0, 2*np.pi, 50)[np.newaxis]
    x = (r*np.cos(theta)).reshape(-1)
    y = (r*np.sin(theta)).reshape(-1)
    z = np.sin(-x*y)
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
