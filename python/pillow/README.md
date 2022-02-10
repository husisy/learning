# Pillow

1. link
   * [official-site](https://python-pillow.org/)
   * [documentation](https://pillow.readthedocs.io/en/stable/)
   * [github](https://github.com/python-pillow/Pillow)
   * [Handbook](https://pillow.readthedocs.io/en/5.0.0/handbook/index.html)
2. install
   * `conda install -c conda-forge pillow`
   * `pip install Pillow`
   * `from PIL import Image`：pillow是package name（对于setuptools而言），PIL是module name（对于Python而言），一个package中可以包含多个module
   * `Pillow` and [PIL](https://pypi.org/project/PIL/) **不能**共存
3. PIL: Python Image Library
4. `Image`: Image对象，使用Image.new、Image.fromarray、Image.frombytes、Image.frombuffer等方法构造。提供对图像简单处理的方法（如：convert、copy、crop、filter、resize、rotate、save、show、transform等）和图像的基本属性/信息。
   * `PIL.Image.new(mode, size, color=0)`中color参数
   * 16进制的颜色表示，如#rgb, #rrggbb
   * rgb函数，如rgb(255,0,0), rgb(100%, 0%, 100%)
   * HSL（色相、饱和、亮度）函数，如hsl(0,100%,100%），色相取值为0到360，饱和度和亮度取值为0%到100%
   * 常规的HTML颜色名字，如red, Red
5. `ImageChops` (CHannel OPerations)，提供对图像颜色通道的算术运算
6. `ImageColor`: 包含颜色表和从CSS3表示的颜色到RGB颜色的转换器，用于构造Image对象
