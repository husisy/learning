import os
import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# plt.ion()

def demo_NOTO_font():
    ## macos
    # brew tap homebrew/cask-fonts
    # brew install font-noto-sans-cjk-sc font-noto-serif-cjk-sc
    # brew ls font-noto-sans-cjk-sc
    ## ubuntu
    # sudo apt install fonts-noto-cjk
    # dpkg-query -L fonts-noto-cjk
    # https://albertauyeung.github.io/2020/03/15/matplotlib-cjk-fonts.html/
    # https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-cjk-fonts
    system = platform.system()
    if system=='Darwin':
        font_dirs = [os.path.expanduser('~/Library/Fonts')]
    elif system=='Linux':
        font_dirs = ['/usr/share/fonts/opentype/noto']
    else:
        raise ValueError('Unknown system: {}'.format(system))
    font_files = matplotlib.font_manager.findSystemFonts(fontpaths=font_dirs)
    for x in font_files:
        matplotlib.font_manager.fontManager.addfont(x)
    # [f.name for f in matplotlib.font_manager.fontManager.ttflist if ('Noto' in f.name) and ('CJK' in f.name)]
    if system=='Darwin':
        matplotlib.rcParams['font.family'] = ['Noto Sans CJK SC']
    elif system=='Linux':
        matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP'] #no idea

    fig, ax = plt.subplots()
    ax.set_xlim(-1.5,0.5)
    ax.set_ylim(-0.5,1.5)
    ax.set_aspect('equal')
    ax.text(0, 0, '你好\n世界', fontsize=40, horizontalalignment='center', verticalalignment='center')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_latex_amsmath():
    # matplotlib.verbose.level = 'debug-annoying'
    # https://stackoverflow.com/q/46259617/7290857
    # https://github.com/matplotlib/matplotlib/issues/22166
    matplotlib.rcParams['text.usetex'] = False #default=False (mathtext mode)
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    fig,ax = plt.subplots()
    ax.set_xlabel(r'$E=mc^2$')
    ax.set_title(r'$\lVert X \rVert_2$')
