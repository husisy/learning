# meson

1. link
   * [github/meson](https://github.com/mesonbuild/meson)
   * [documentation](https://mesonbuild.com/SimpleStart.html)
2. install [meson/docs](https://mesonbuild.com/SimpleStart.html)
   * `meson --version`
   * (or) `mamba install -c conda-forge meson`
   * (or) `pip install meson`
3. configure step and build stepe

```bash
meson --version

mkdir tbd233
cd tbd233
meson init --name testproject --build
build/testproject
cd build && meson compile
```
