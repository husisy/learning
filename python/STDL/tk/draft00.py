from tkinter import Frame, Label, Button, Entry
import tkinter.messagebox as messagebox


class MyApplication(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.helloLabel = Label(self, text='Hello, world!')
        self.helloLabel.pack()
        self.quitButton = Button(self, text='Quit', command=self.quit)
        self.quitButton.pack()

        self.nameInput = Entry(self)
        self.nameInput.pack()

        self.alertButton = Button(self, text='Hello', command=self._hello)
        self.alertButton.pack()

    def _hello(self):
        tmp1 = 'hello, {}'.format(self.nameInput.get())
        messagebox.showinfo('message', tmp1)

app = MyApplication()
app.master.title('Hello World')
app.mainloop()
