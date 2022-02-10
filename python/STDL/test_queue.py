import queue
import numpy as np


def test_queue_order(N0=5)

    data_ = np.random.randint(0, 1000, size=N0).tolist()

    q0 = queue.Queue()
    for x in data_:
        q0.put(x)
    data0 = []
    while not q0.empty():
        data0.append(q0.get())
    assert all(x==y for x,y in zip(data_,data0))

    q1 = queue.LifoQueue()
    for x in data_:
        q1.put(x)
    data0 = []
    while not q1.empty():
        data0.append(q1.get())
    assert all(x==y for x,y in zip(data_,data0[::-1]))

# queue.PriorityQueue
