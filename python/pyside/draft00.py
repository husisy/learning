import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

def demo00():
    app = QtWidgets.QApplication(sys.argv)
    # label = QtWidgets.QLabel("Hello World!")
    label = QtWidgets.QLabel("<font color=red size=40>Hello World!</font>") # html
    label.show()
    app.exec()


@QtCore.Slot()
def _demo01_say_hello():
    print("Button clicked, Hello!")

def demo01():
    app = QtWidgets.QApplication(sys.argv)
    button = QtWidgets.QPushButton("Click me")
    button.clicked.connect(_demo01_say_hello)
    button.show()
    app.exec()


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World", alignment=QtCore.Qt.AlignCenter)
        self.button.clicked.connect(self.magic)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))


def demo02():
    app = QtWidgets.QApplication([])
    widget = MyWidget()
    widget.resize(300, 200)
    widget.show()
    app.exec()


class Communicate(QtCore.QObject):
    # create two new signals on the fly: one will handle int type, the other will handle strings
    speak = QtCore.Signal((int,), (str,))

    def __init__(self, parent=None):
        super().__init__(parent)

        self.speak[int].connect(self.say_something)
        self.speak[str].connect(self.say_something)

    # define a new slot that receives a C 'int' or a 'str' and has 'say_something' as its name
    @QtCore.Slot(int)
    @QtCore.Slot(str)
    def say_something(self, arg):
        if isinstance(arg, int):
            print("This is a number:", arg)
        elif isinstance(arg, str):
            print("This is a string:", arg)


def demo03():
    app = QtWidgets.QApplication(sys.argv)
    someone = Communicate()
    # emit 'speak' signal with different arguments. we have to specify the str as int is the default
    someone.speak.emit(10)
    someone.speak[str].emit("Hello everybody!")


class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("My Form")

        self.edit = QtWidgets.QLineEdit("Write my name here..")
        self.button = QtWidgets.QPushButton("Show Greetings")
        self.button.clicked.connect(self.greetings)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)

    def greetings(self):
        print(f"Hello {self.edit.text()}")


def demo04():
    app = QtWidgets.QApplication(sys.argv)
    form = Form()
    form.show()
    sys.exit(app.exec())


def get_rgb_from_hex(code):
    code_hex = code.replace("#", "")
    rgb = tuple(int(code_hex[i:i+2], 16) for i in (0, 2, 4))
    ret = QtGui.QColor.fromRgb(rgb[0], rgb[1], rgb[2])
    return ret


def demo05():
    colors = [("Red", "#FF0000"), ("Green", "#00FF00"), ("Blue", "#0000FF"), ("Black", "#000000"),
            ("White", "#FFFFFF"), ("Electric Green", "#41CD52"), ("Dark Blue", "#222840"), ("Yellow", "#F9E56d")]
    app = QtWidgets.QApplication()
    table = QtWidgets.QTableWidget()
    table.setRowCount(len(colors))
    table.setColumnCount(len(colors[0]) + 1)
    table.setHorizontalHeaderLabels(["Name", "Hex Code", "Color"])
    for i, (name, code) in enumerate(colors):
        item_name = QtWidgets.QTableWidgetItem(name)
        item_code = QtWidgets.QTableWidgetItem(code)
        item_color = QtWidgets.QTableWidgetItem()
        item_color.setBackground(get_rgb_from_hex(code))
        table.setItem(i, 0, item_name)
        table.setItem(i, 1, item_code)
        table.setItem(i, 2, item_color)
    table.show()
    app.exec()

if __name__ == "__main__":
    # run one of the demos
    # demo00()
    # demo01()
    # demo02()
    # demo03()
    # demo04()
    demo05()
