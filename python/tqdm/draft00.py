import time
import random
from tqdm import tqdm


def hf0(*args, **kwargs):
    time.sleep(0.03)


for ind0 in tqdm(range(100)):
    hf0(ind0)


with tqdm(total=100) as pbar:
    for ind0 in range(100):
        hf0(ind0)
        pbar.update(1)


with tqdm(total=100) as pbar:
    for ind0 in range(100):
        hf0()
        pbar.set_description('GEN %i' % ind0) #left
        pbar.set_postfix(loss=random.random(), gen=random.randint(1,999), str='h') #right
        pbar.update(1)


with tqdm(total=10, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
          postfix=["Batch", dict(value=0)]) as pbar:
    for i in range(100):
        hf0()
        pbar.postfix[1]["value"] = i / 2
        pbar.update()
