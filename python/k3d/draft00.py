import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import k3d

def k3d_surface(z, xmin=0, xmax=1, ymin=0, ymax=1, colormap_name='cool'):
    z = z.astype(np.float32)
    colormap = getattr(matplotlib.cm, colormap_name)
    tmp0 = [colormap(x) for x in range(colormap.N)]
    tmp1 = np.linspace(0, 1, num=len(tmp0), endpoint=True)
    k3d_colormap = [float(z) for x,y in zip(tmp1,tmp0) for z in (x,y[0],y[1],y[2])]
    hsurf = k3d.surface(z, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                attribute=z, color_map=k3d_colormap, flat_shading=False)
    return hsurf


xmin, xmax, ymin, ymax = -3, 3, 0, 3
x = np.linspace(xmin, xmax, 50)
y = np.linspace(ymin, ymax, 60)
f = np.sin(x**2 + y[:,np.newaxis]**2)
plot = k3d.plot()
plot += k3d_surface(f, xmin=x.min(), xmax=x.max(), ymin=y.min(), ymax=y.max())
plot.display()

x = np.linspace(-5, 5)
y = np.linspace(-5, 5)
z0 = np.sin(np.sqrt(x**2 + y[:,np.newaxis]**2))
z1 = np.cos(np.sqrt(x**2 + y[:,np.newaxis]**2))

plot = k3d.plot()
plot += k3d_surface(z0, xmin=x.min(), xmax=x.max(), ymin=y.min(), ymax=y.max())
plot.display()

plot = k3d.plot()
plot += k3d_surface(z1, xmin=x.min(), xmax=x.max(), ymin=y.min(), ymax=y.max())
plot.display()
