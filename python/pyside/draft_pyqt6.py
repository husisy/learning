import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout

# https://www.pythonguis.com/tutorials/pyqt6-creating-your-first-window/

def demo00():
    app = QApplication(sys.argv)
    window = QWidget()
    window.show()
    app.exec()
    # python draft00.py


class MainWindow01(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        # self.setFixedSize(QSize(400, 300)) #width=400pixel, height=300pixel
        button = QPushButton("Press Me!")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)
        button.clicked.connect(self.the_button_was_toggled)
        # button.setChecked(True) #default to False
        self.setCentralWidget(button) # Set the central widget of the Window.

    def the_button_was_clicked(self):
        print("Clicked!")

    def the_button_was_toggled(self, checked):
        print("Checked?", checked)

def demo01():
    app = QApplication(sys.argv)
    window = MainWindow01()
    window.show()
    app.exec()


class MainWindow02(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.label = QLabel()
        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)
        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

def demo02():
    app = QApplication(sys.argv)
    window = MainWindow02()
    window.show()
    app.exec()


class AnotherWindow03(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel("Another Window")
        layout.addWidget(self.label)
        self.setLayout(layout)


class MainWindow03(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button = QPushButton("Push for Window")
        self.button.clicked.connect(self.show_new_window)
        self.setCentralWidget(self.button)
        self.w = None

    def show_new_window(self, checked):
        if self.w is None:
            self.w = AnotherWindow03()
        self.w.show()


def demo03():
    app = QApplication(sys.argv)
    window = MainWindow03()
    window.show()
    app.exec()


if __name__ == '__main__':
    # demo00()
    # demo01()
    # demo02()
    demo03()
