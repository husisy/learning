# mathematica

1. link
   * official site, [link](http://www.wolfram.com/)
   * [documentation](https://reference.wolfram.com/language/)
   * [wolframscript](https://www.wolfram.com/wolframscript/)
   * [wolframscript-documentation](https://reference.wolfram.com/language/ref/program/wolframscript.html)
   * [wolfram-engine-for-developer-documentation](https://www.wolfram.com/engine/)
   * [jupyter-notebook](https://github.com/WolframResearch/WolframLanguageForJupyter)
2. 安装wolfram script
   * 下载`WolframScript_12.2.0_LINUX64_amd64_CN.deb`
   * `sudo dpkg -i WolframScript_12.2.0_LINUX64_amd64_CN.deb`
3. 安装wolfram-engine-for-developer：安装engine之后必须安装wolframscript
4. 特殊变量：`$ScriptCommandLine`, `$UserBaseDirectory`

```mathematica
wolframscript -code 2+2
wolframscript -code 'Print["hello world"]'
wolframscript -code 'StringReverse["hello"]'
wolframscript -code 'Graphics3D[Sphere[]]' -format PNG > tbd00.png
```

TODO

1. [ ] 把`draft_xx.nb`转换为`draft_xx.wls`或者`draft_xx.ipynb`

## minmum working example

`draft00.wls`

```mathematica
Print["hello world"];
```

运行：`wolframscript -file draft00.wls`

## misc

1. 转移电脑license [mma-doc](https://support.wolfram.com/12412?src=mathematica) hwo do i move wolfram software to a new pc
