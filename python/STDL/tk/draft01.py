'''
https://github.com/Akuli/tkinter-tutorial
https://www.tcl.tk/man/tcl/TkCmd/contents.htm
'''
import time
import tkinter
from tkinter import Tk, ttk # themed tk


def tkinter_label():
    app = Tk()
    tkinter.Label(app, text='close windows').pack()

    x1 = tkinter.Label(app)
    x1['text'] = 'one by one'
    x1.pack()

    x2 = tkinter.Label(app)
    x2.config(text='to go through')
    x2.pack()

    x3 = tkinter.Label(app)
    x3.configure(text='this tutorial')
    x3.pack()
    app.mainloop()


def tk_vs_ttk():
    app = Tk()
    tkinter.Button(app, text='click me by tkinter.Label()',
                   command=lambda: print('tk_button')).pack()
    ttk.Button(app, text='click me by ttk.Label()',
               command=lambda: print('ttk_button')).pack()
    app.mainloop()


def ttk_frame():
    # always recommand to do this: https://github.com/Akuli/tkinter-tutorial/blob/master/getting-started.md#hello-ttk
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='always use ttk.Frame() to packing widget').pack()
    # app.mainloop()
    app.update()
    time.sleep(2)
    app.destroy()


def ttk_app_property():
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand=True)
    app.title('this is a title')
    app.geometry('200x100')
    app.minsize(100, 50)
    app.maxsize(400, 200)
    # app.resizable(False, False)
    app.mainloop()


def ttk_geometry_pack():
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand=True)

    # order matters
    x3 = ttk.Label(
        frame, text='label, relief=sunken, side=bottom, fill=x', relief='sunken')
    x3.pack(side='bottom', fill='x')
    # side=
    #  left:
    #  right:
    #  [top]:
    #  bottom:
    # fill=
    #  x: fill all of the space horizontally
    #  y: fill all of the space vertically
    #  both:

    x1 = ttk.Button(frame, text='button, side=left, fill=both, expand=True')
    x1.pack(side='left', fill='both', expand=True)

    x2 = ttk.Label(frame, text='label, size=left')
    x2.pack(side='left')

    app.geometry('300x150')
    app.mainloop()


def ttk_geometry_grid():
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand='True')

    ttk.Button(frame, text='row=0,column=0,nswe').grid(
        row=0, column=0, sticky='nswe')
    ttk.Label(frame, text='row=0,column=1').grid(row=0, column=1)
    ttk.Label(frame, text='row=1,column=1,columnspan=2,we', relief='sunken').grid(
        row=1, column=0, columnspan=2, sticky='we')
    # sticky=
    #  nswe: equivalent to fill=both
    #  we: equivalent to fill=x
    #  ns: equivalent to fill=y
    #  []

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    app.geometry('300x150')
    app.mainloop()


def ttk_calculator():
    '''https://github.com/Akuli/tkinter-tutorial/blob/master/geometry-managers.md#grid
    ,-------------------.
    | Calculator    | X |
    |-------------------|
    | 7 | 8 | 9 | * | / |
    |---+---+---+---+---|
    | 4 | 5 | 6 | + | - |
    |---+---+---+-------|
    | 1 | 2 | 3 |       |
    |-------+---|   =   |
    |   0   | . |       |
    `-------------------'
    '''
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand=True)

    rows = [
        ['7', '8', '9', '*', '/'],
        ['4', '5', '6', '+', '-'],
        ['1', '2', '3', None, None],
        [None, None, '.', None, None],
    ]
    for y, row in enumerate(rows):
        for x, character in enumerate(row):
            if character is not None:
                ttk.Button(frame, text=character, width=3).grid(
                    row=y, column=x, sticky='nswe')

    ttk.Button(frame, text='0', width=1).grid(
        row=3, column=0, columnspan=2, sticky='nswe')
    ttk.Button(frame, text='=', width=1).grid(
        row=2, column=3, rowspan=2, columnspan=2, sticky='nswe')

    for x in range(5):
        frame.grid_columnconfigure(x, weight=1)
    for y in range(4):
        frame.grid_rowconfigure(y, weight=1)

    app.title('Calculator')
    app.mainloop()


def ttk_place():
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand=True)

    ttk.Label(frame, text='ttk.label().place(relx=0.5,rely=0.3)').place(
        relx=0.5, rely=0.3, anchor='center')
    ttk.Button(frame, text='quit', command=app.destroy).place(
        relx=0.5, rely=0.8, anchor='center')
    app.geometry('250x150')
    app.mainloop()


if __name__ == "__main__":
    # tkinter_label()
    # tk_vs_ttk()
    ttk_frame()
    # ttk_app_property()
    # ttk_geometry_pack()
    # ttk_geometry_grid()
    # ttk_calculator()
    # ttk_place()
