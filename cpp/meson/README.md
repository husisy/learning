# meson

1. link
   * [github/meson](https://github.com/mesonbuild/meson)
   * [documentation](https://mesonbuild.com/SimpleStart.html)
2. install [meson/docs](https://mesonbuild.com/SimpleStart.html)
   * `mamba install meson`
   * `pip install meson`
3. configure step and build stepe

```bash
meson --version
meson setup build
meson compile #in the build directory
# --buildtype=debugoptimized
meson test #in the build directory

```

## MWE00

```bash
ws00-c/
├── meson.build
└── main.c
```

```bash
# meson.build
project('tutorial', 'c')
executable('demo', 'main.c')
```

```c
// main.c
#include <stdio.h>
int main(int argc, char **argv) {
  printf("Hello there.\n");
  return 0;
}
```

```bash
# create build directory
meson setup build
cd build
meson compile
./demo
```

## MWE01

```bash
mkdir tbd00
cd tbd00
meson init --name testproject --build
build/testproject
cd build && meson compile
```
