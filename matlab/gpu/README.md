# GPU computing

1. link
   * [documentation](https://www.mathworks.com/help/parallel-computing/gpu-computing.html)

旧版本matlab

```bash
# https://www.mathworks.com/matlabcentral/answers/289457-parallel-gpu-cudakernel-slow-on-gtx-1080
# https://www.mathworks.com/matlabcentral/answers/289457-parallel-gpu-cudakernel-slow-on-gtx-1080
# https://www.mathworks.com/matlabcentral/answers/79275-gpudevice-command-very-slow
export CUDA_CACHE_MAXSIZE=2147483647
export CUDA_CACHE_DISABLE=0
```

gpu-benchmark

```MATLAB
% https://www.mathworks.com/matlabcentral/answers/464145-running-code-on-gpu-seems-much-slower-than-doing-so-on-cpu
% CPU data
a = rand([100, 100], 'single');
h = rand([100, 100], 'single');
% GPU data
aa = gpuArray(a);
hh = gpuArray(h);
% Measuring CONV2 with one output
cpuTime = timeit(@() conv2(h, a, 'full'), 1);
gpuTime = gputimeit(@() conv2(hh, aa, 'full'), 1);
```
