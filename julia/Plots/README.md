# Plots

1. link
   * [documentation](http://docs.juliaplots.org/latest/)
   * [github](https://github.com/JuliaPlots/Plots.jl)
2. install
   * `]add Plots`
   * `]add GR` for speed
   * `]add PlotlyJS` `]add ORCA` for interactivity
   * `]add PyPlot` otherwise
3. `using Plots`
4. keyword alias `c/color m/marker`
5. global variable `Plots.CURRENT_PLOT`
6. input parameter rule
   * `plot(x,y,z)`: 3-dimensional data for 3D plot
   * `plot(x,y,attribute=value)`: 2-dimensional with an attribute
7. 测试发现`Plots`暂不支持x11 server
