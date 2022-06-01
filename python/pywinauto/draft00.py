import subprocess
import pywinauto

# from pywinauto.application import Application

app = pywinauto.application.Application(backend="uia").start("notepad.exe")
app.UntitledNotepad.type_keys("%FX")

app = pywinauto.application.Application(backend="uia").start('notepad.exe')
dlg_spec = app.UntitledNotepad
actionable_dlg = dlg_spec.wait('visible')


# binding to an exist GUI
subprocess.Popen('calc.exe', shell=True)
dlg = pywinauto.Desktop(backend="uia").Calculator
dlg.wait('visible')
dlg.type_keys("233")
