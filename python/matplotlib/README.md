# matplotlib

1. link
   * [github](https://github.com/matplotlib/matplotlib)
   * [documentation](https://matplotlib.org/index.html)
   * [coding-style](https://matplotlib.org/stable/users/explain/quick_start.html#coding-styles)
   * [matplotlib/gallery](https://matplotlib.org/stable/gallery/index.html)
   * [github/scienceplots](https://github.com/garrettj403/SciencePlots)
2. setup
   * `mamba install matplotlib`
   * `pip install matplotlib`
3. figure part: figure, axes, title, legend, line, marker, grid, spline, axis label, major/minor tick, major/minor tick label, artist
4. 偏见
   * **禁止**使用`matplotlib.pylab`：初衷模仿MATLAB，包含`pyplot`和`numpy`模块中大部分的接口
   * 尽量不使用`plt` state machine environment api，转为使用 objec orientied interface
   * 关闭图片窗格**必须**显示调用`plt.close(plt.gcf())`, see [link-last-paragraph](https://matplotlib.org/tutorials/introductory/pyplot.html#working-with-multiple-figures-and-axes)
   * **禁止**在`matplotlib`中使用`numpy.recarray`, `pandas.DataFrame`，虽然支持，see [link](https://matplotlib.org/tutorials/introductory/pyplot.html#plotting-with-keyword-strings)
5. configuration
   * `matplotlib.rcParams`
   * `matplotlibrc`
   * `matplotlib.get_configdir()`
   * `*.mplstyle`: style configuration
   * math: mathtext, LaTeX [matplotlib/doc](https://matplotlib.org/stable/users/explain/text/mathtext.html)
6. backend
   * interactive backend (user interface backend): pygtk, wxpython, tkinter, qt4, macosx
   * non-interactive backend (hardcopy backend): PNG, SVG, PDF, PS
   * `matplotlib.rcParams['backend']`, `os.environ['MPLBACKEND']`, `matplotlib.use()`
   * renderer (vector or raster) and canvas
   * case-insensitive: `gtk3agg` or `GTK3Agg`
   * Anti-Grain Geometry C++ library (Agg)
7. artist: line, text (LaTeX expression), histogram, annotation

```Python
import matplotlib
import matplotlib.pyplot as plt

plt.ion()
plt.ioff()
plt.draw()
matplotlib.interactive()
matplotlib.is_interactive()
plt.close()
plt.clf()
plt.cla()
ax.remove()
```

## Scienceplots

1. link
   * [github](https://github.com/garrettj403/SciencePlots)
   * [scienceplots/CJK-font-install](https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-cjk-fonts)
2. install
   * `mamba install -c conda-forge scienceplots`
   * `pip install SciencePlots`
