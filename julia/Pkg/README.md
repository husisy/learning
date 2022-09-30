# Pkg

1. link
   * [documentation/Pkg](https://docs.julialang.org/en/v1/stdlib/Pkg/)
2. 在julia命令行下按键`]`进入Pkg命令行模式，按键`backwpspace`退出Pkg命令行模式
3. `pkg>`中的常用命令
   * `add JSON StaticArrays`
   * `rm JSON StaticArrays`
   * `add Example`
   * `add Example@0.4`
   * `add Example#master`
   * `add https://github.com/JuliaLang/Example.jl`
   * `rm Example`
   * `free Example`
   * `update Example`
   * `update`
   * `?`, `?develop`
   * `test Example`
4. environenment
   * `activate tutorial`
   * `activate` return to the default environment
   * `status`, `status --manifest`
   * `add Example`
   * `develop --local Example`
5. project
   * `activate .`
6. create package
   * `generate HelloWorld`
   * `cd("HelloWorld")`
   * `pkg> activate .`
   * `import HelloWorld; HelloWorld.greet()`
7. TODO
   * [PkgTemplates](https://github.com/invenia/PkgTemplates.jl)
