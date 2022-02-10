# oommf

1. link
   * [github/fangohr/oommf](https://github.com/fangohr/oommf)
   * [oommf-official-site](https://math.nist.gov/oommf/oommf.html)
   * [oommf-tutorial](https://math.nist.gov/oommf/oommf_tutorial/tutorial.html)
   * [oommf-user-guide](https://math.nist.gov/oommf/doc/)
2. installation
   * windows：下载安装包即可
   * linux：见下方代码块
3. 运行 `tclsh oommf.tcl`
4. concept
   * `Oxsii`

```bash
git clone [xxx](https://github.com/fangohr/oommf.git)
sudo apt-get install tcl-dev tk-dev
export OOMMF_TCL_INCLUDE_DIR="/usr/include/tcl8.6"
export OOMMF_TK_INCLUDE_DIR="/usr/include/tk"
make build
```

## working example

nanoHUB example from oommf tutorial

1. link
   * [nanohub](https://nanohub.org/tools/oommf/)
   * [instruction](https://nanohub.org/resources/23834)
