# julia language

1. link
   * [official site](https://julialang.org/)
   * [documentation](https://docs.julialang.org/en/v1/)
   * [github](https://github.com/JuliaLang/julia)
   * [style guide](https://docs.julialang.org/en/v1/manual/style-guide/)
   * [Roger-luo-julia编程指南](https://github.com/Roger-luo/Brochure.jl)
   * [语义化版本](https://semver.org/lang/zh-CN/)
   * [wikibook-introducing julia](https://en.wikibooks.org/wiki/Introducing_Julia)
2. 安装（临时添加至全局路径）
   * **请**坐和放宽，网络问题急也没用
   * linux-bash: `export PATH="$PATH:/PATH/TO/julia-1.3.1/bin"`
   * win-powershell: `$env:Path += ";/PATH/TO/Julia-1.3.1/bin"`
   * 镜像`] registry add https://mirrors.ustc.edu.cn/julia/registries/General.git`
   * 镜像`] registry add https://mirrors.zju.edu.cn/julia/registries/General.git`
3. REPL (Read-Eval-Print-Loop)
4. 特殊变量
   * `ans`：仅限于交互式环境
   * `ARGS`
   * `PROGRAM_FILE`
5. 常用函数：`exit() print() show() println() typeof() bitstring() eps() nextfloat() prevfloat() zero() one()`
6. scripts `julia draft00.jl 2 23 233`
   * `julia -e 'for x in ARGS; println(x); end' 2 23 233`
7. 配置文件 `~/.julia/config/startup.jl`
8. misc
   * `Sys.WORD_SIZE`
   * `DivideError`: `1/0`, `typemin()/-1`, `rem(1,0)`, `mod(1,0)`
   * arbitrary precision arithmetic
9. 个人偏见
    * **禁止**使用juxtaposed literal coefficient syntax。例外：`1 + 2im`
    * **禁止**使用`using`：其语义是混淆的（非用户可控的破坏当前命名空间），其语义是完全可被`import`替代的（限制method extension这点没有意义）
10. 数据类型`Char String Tuple`
    * non-standard string literals: regular expressions, byte array, version number, raw string
    * named tuple
11. function
    * assignment form
    * type declarations
    * operator
    * generic function

## indexing

1. `[0] [end]`
2. `firstindex() lastindex()`

## misc00

```julia
include("xxx.jl")
# ] add Revise
```

TODO list

1. `] add Revise` 热加载
   * 将`using Revise`添加到`startup.jl`
2. package: `Revise Plots GraphRecipes DataFrames.jl Flux`
3. 数组
4. 表达式
5. is `3.5:9.5` stable for float number
