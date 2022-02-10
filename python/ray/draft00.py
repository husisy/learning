import time

import ray
ray.init()

@ray.remote
def f(x):
    return x * x

futures = [f.remote(x) for x in range(4)]
print(ray.get(futures)) # [0, 1, 4, 9]


@ray.remote
class Counter:
    def __init__(self):
        self.n = 0

    def increment(self):
        time.sleep(5)
        self.n += 1

    def read(self):
        return self.n

t0 = time.time()
counters = [Counter.remote() for _ in range(4)]
[x.increment.remote() for x in counters]
futures = [x.read.remote() for x in counters]
print(ray.get(futures)) # [1, 1, 1, 1]
print('[zcinfo]', time.time()-t0)
