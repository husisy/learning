# multithreading select poll
import os
import numpy as np
import time
import select
import threading


class DummyThreadRunner01(threading.Thread):
    def __init__(self, pipe_r_list):
        super().__init__()
        self.pipe_r_list = pipe_r_list
    def run(self):
        while True:
            pipe_r, _, _ = select.select(self.pipe_r_list, [], [])
            pipe_r = pipe_r[0]
            print('[pipe]', pipe_r)
            x = os.read(pipe_r, 1024)
            print('[content]', x)


N0 = 4
pipe_r_list = []
pipe_w_list = []
for _ in range(N0):
    x = os.pipe()
    pipe_r_list.append(x[0])
    pipe_w_list.append(x[1])
print(list(zip(pipe_r_list, pipe_w_list)))

worker = DummyThreadRunner01(pipe_r_list)
worker.start()


for x in np.random.randint(N0, size=(5)):
    tmp0 = f'write to {x}'
    print('[main]', pipe_w_list[x])
    os.write(pipe_w_list[x], tmp0.encode('utf-8'))
    time.sleep(0.01)
