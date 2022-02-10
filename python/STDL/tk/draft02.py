import tkinter
from tkinter import Tk, ttk


def make_calculator_frame(frame):
    rows = [
        ['7', '8', '9', '*', '/'],
        ['4', '5', '6', '+', '-'],
        ['1', '2', '3', None, None],
        [None, None, '.', None, None],
    ]
    for y, row in enumerate(rows):
        for x, character in enumerate(row):
            if character is not None:
                ttk.Button(frame, text=character, width=3).grid(row=y, column=x, sticky='nswe')

    ttk.Button(frame, text='0', width=1).grid(row=3, column=0, columnspan=2, sticky='nswe')
    ttk.Button(frame, text='=', width=1).grid(row=2, column=3, rowspan=2, columnspan=2, sticky='nswe')

    for x in range(5):
        frame.grid_columnconfigure(x, weight=1)
    for y in range(4):
        frame.grid_rowconfigure(y, weight=1)

    return frame


def make_message_frame(frame):
    ttk.Label(frame, text='This is a very important message.').place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(frame, text='OK').place(relx=0.5, rely=0.8, anchor='center')
    return frame


if __name__ == '__main__':
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand=True)

    ttk.Label(frame, text='This is a status bar.', relief='sunken').pack(side='bottom', fill='x')

    calculator_frame = ttk.Frame(frame)
    make_calculator_frame(calculator_frame).pack(side='left', fill='y')

    message_frame = ttk.Frame(frame)
    make_message_frame(message_frame).pack(side='left', fill='both', expand=True)

    app.geometry('450x200')
    app.minsize(400, 100)
    app.mainloop()
