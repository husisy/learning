# msys2

1. link
   * [official site](http://www.msys2.org/)
   * [mingw-w64](https://mingw-w64.org/doku.php/start)
2. update: `pacman -Syuu`
3. [package](https://github.com/msys2/msys2/wiki/Using-packages)
   * mingw32: `mingw-w64-i686-xxx`
   * mingw64: `mingw-w64-x86_64-xxx`
   * search: `pacman -Ss gcc`
   * install: `pacman -S mingw-w64-x86_64-gcc`
   * remove: `pacman -R mingw-w64-x86_64-gcc`
