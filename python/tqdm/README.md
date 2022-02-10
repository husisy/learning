# tqdm

1. link
   * [github](https://github.com/tqdm/tqdm)
   * [documentation](https://tqdm.github.io/)
2. `from tqdm import trange, tqdm`

```Python
z1 = [math.sin(x) for x in tqdm(range(1000000))]

N1 = round(1e6)
list1 = [None]*N1
with tqdm(total=N1) as pbar:
    for ind1 in range(N1):
        pbar.update(1)
        list1[ind1] = math.sin(ind1)
```
