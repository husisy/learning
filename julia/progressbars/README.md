# ProgressBars.jl

1. link
   * [github](https://github.com/cloud-oak/ProgressBars.jl)
2. install `] add ProgressBars`

```julia
using ProgressBars
for i in ProgressBar(1:100)
    sleep(0.01)
end
```
