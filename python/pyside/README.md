# pyside

1. link
   * [documentation](https://doc.qt.io/qtforpython-6/index.html)
2. install
   * `pip install pyside6`
   * `mamba install -c conda-forge pyside6`
3. concept
   * Qt for python
   * Qt Widget application
   * Qt Quick application (QML language, Javascript)
   * Shiboken
   * Shiboken generator
   * signal and slot
   * model/view

```Python
import PySide6.QtCore
print(PySide6.__version__)
print(PySide6.QtCore.__version__) #Qt version used to compile PySide6
```

```bash
export QT_LOGGING_RULES="qt.pyside.libpyside.warning=true"
```

## pyqt

1. link
   * [Qt5/documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/introduction.html)
   * [pythongui/tutorials](https://www.pythonguis.com/pyqt6/)
2. install
   * `pip install PyQt6`
   * not available for conda
