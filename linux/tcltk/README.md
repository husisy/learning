# Tcl Tk

## Tcl

1. link
   * [tcltk-documentation](https://www.tcl.tk/doc/)
   * [tcltk-wiki](https://wiki.tcl-lang.org/welcome)
2. ubuntu installation
   * `sudo apt install tcl-dev tk-dev`
3. start `tclsh draft00.tcl`
4. built-in变量
   * `tcl_precision`
   * `tcl_version`
5. 偏见
   * 花括号之间必有有空格`{} {}`

## Tk

1. link
   * [documentation](https://tkdocs.com/)
   * [tutorial](https://tkdocs.com/tutorial/index.html)
2. widget: entry, label, frame, checkbox, tree view, scrollbar, text area

```Tik
package require Tk
grid [ttk::button .b -text "hello world"]
```
