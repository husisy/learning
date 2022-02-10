import tkinter
from tkinter import Tk, ttk, messagebox, filedialog, simpledialog, colorchooser


def tk_app_frame():
    app = Tk()
    frame = ttk.Frame(app)
    frame.pack(fill='both', expand=True)
    return app, frame


def tk_messagebox(state='showinfo'):
    # https://www.tcl.tk/man/tcl/TkCmd/messageBox.htm
    # default: return value when popup-windows is closed without picking
    def hf_showinfo():
        messagebox.showinfo('showinfo', 'info: hello world')
    def hf_showwarning():
        messagebox.showwarning('showwarning', 'this is a warning')
    def hf_showerror():
        messagebox.showerror('showerror', 'this is an error')
    def hf_askyesno():
        ret = messagebox.askyesno('askyesno', 'pick one') #True/[False]
        print(type(ret), ret)
    def hf_askyesnocancel():
        ret = messagebox.askyesnocancel('askyesnocancel', 'pick one') #True/False/[None]
        print(type(ret), ret)
    def hf_askokcancel():
        ret = messagebox.askokcancel('askokcancel', 'pick one') #True/[False]
        print(type(ret), ret)
    def hf_askretrycancel():
        ret = messagebox.askretrycancel('askretrycancel', 'pick one') #True/[False]
        print(type(ret), ret)
    def hf_askquestion():
        ret = messagebox.askquestion('askquestion', 'pick one') #'yes'/'no', NO DEFAULT
        print(type(ret), ret)
    tmp1 = {
        'showinfo': hf_showinfo,
        'showwarning': hf_showwarning,
        'showerror': hf_showerror,
        'askyesno': hf_askyesno,
        'askyesnocancel': hf_askyesnocancel,
        'askokcancel': hf_askokcancel,
        'askretrycancel': hf_askretrycancel,
        'askquestion': hf_askquestion,
    }
    app,frame = tk_app_frame()
    ttk.Button(frame, text='messagebox: click me', command=tmp1[state]).pack()
    app.geometry('400x300')
    app.mainloop()

def tk_filedialog(state='askopenfilename'):
    # https://www.tcl.tk/man/tcl/TkCmd/getOpenFile.htm
    # https://www.tcl.tk/man/tcl/TkCmd/chooseDirectory.htm
    def hf_askopenfilename():
        ret = filedialog.askopenfilename(title='askopenfilename') #[str]
        print(type(ret), ret)
    def hf_asksaveasfilename():
        # ret = filedialog.asksaveasfile(title='asksaveasfile') #TextIOWrapper
        ret = filedialog.asksaveasfilename(title='asksaveasfilename') #[str]
        print(type(ret), ret)
    tmp1 = {
        'askopenfilename': hf_askopenfilename,
        'asksaveasfilename': hf_asksaveasfilename,
    }
    app,frame = tk_app_frame()
    ttk.Button(frame, text='filedialog: click me', command=tmp1[state]).pack()
    app.geometry('400x300')
    app.mainloop()

def tk_simpledialog(state='askfloat'):
    # http://effbot.org/tkinterbook/tkinter-dialog-windows.htm
    def hf_askfloat():
        ret = simpledialog.askfloat('askfloat', 'input float: ') #float/[None]
        print(type(ret), ret)
    def hf_askinteger():
        ret = simpledialog.askinteger('askinteger', 'input integer: ') #int/[None]
        print(type(ret), ret)
    def hf_askstring():
        ret = simpledialog.askstring('askstring', 'input: ') #str/[None]
        print(type(ret), ret)
    tmp1 = {
        'askfloat': hf_askfloat,
        'askinteger': hf_askinteger,
        'askstring': hf_askstring,
    }
    app,frame = tk_app_frame()
    ttk.Button(frame, text='simpledialog: click me', command=tmp1[state]).pack()
    app.geometry('400x300')
    app.mainloop()


def tk_colorchooser(state='askcolor'):
    # https://www.tcl.tk/man/tcl/TkCmd/chooseColor.htm
    def hf_askcolor():
        ret = colorchooser.askcolor(title='askcolor')
        # [(float,float,float),str]/[None,None]
        # ((121.47265625, 114.4453125, 207.80859375), '#7972cf')
        # #7972cf is hexadecimal number in string
        # X11 color names: https://en.wikipedia.org/wiki/X11_color_names
        print(type(ret), ret)
    tmp1 = {
        'askcolor': hf_askcolor,
    }
    app,frame = tk_app_frame()
    ttk.Button(frame, text='colorchooser: click me', command=tmp1[state]).pack()
    app.geometry('400x300')
    app.mainloop()


def tk_show_message_without_app():
    # https://www.tcl.tk/man/tcl/TkCmd/wm.htm
    app = Tk()
    app.withdraw()
    messagebox.showerror('show message without app window', 'this is an error')
    app.deiconify()
    messagebox.showerror('show message with app window', 'this is an error')
    app.destroy()


def tk_custom_dialog():
    '''https://github.com/Akuli/tkinter-tutorial/blob/master/dialogs.md#custom-dialogs'''
    def hf1(app):
        dialog = tkinter.Toplevel()
        frame = ttk.Frame(dialog)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text='custom_diaglog').place(relx=0.5, rely=0.3, anchor='center')

        dialog.transient(app)
        dialog.geometry('300x150')
        dialog.wait_window()

    app,frame = tk_app_frame()
    ttk.Button(frame, text="Click me", command=lambda _x=app: hf1(_x)).pack()
    app.mainloop()


def tk_protocol():
    def hf_wanna_quit(app):
        if messagebox.askyesno('Quit', 'Do you really want to quit? pick one'):
            app.destroy()

    app = tkinter.Tk()
    app.protocol('WM_DELETE_WINDOW', lambda _x=app: hf_wanna_quit(_x))
    app.mainloop()


if __name__ == "__main__":
    tk_messagebox('showinfo')
    tk_messagebox('showwarning')
    tk_messagebox('showerror')
    tk_messagebox('askyesno')
    tk_messagebox('askyesnocancel')
    tk_messagebox('askokcancel')
    tk_messagebox('askretrycancel')
    tk_messagebox('askquestion')

    # tk_filedialog('askopenfilename')
    # tk_filedialog('asksaveasfilename')

    # tk_simpledialog('askfloat')
    # tk_simpledialog('askinteger')
    # tk_simpledialog('askstring')

    # tk_colorchooser('askcolor')

    # tk_show_message_without_app()

    # tk_custom_dialog()

    # tk_protocol()
