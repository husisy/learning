import time
import math
import numpy as np
import joblib

cachedir = 'tbd00'
memory = joblib.Memory(cachedir, verbose=1)
np0 = np.vander(np.arange(3, dtype=np.float64))
@memory.cache
def hf_task(x):
    time.sleep(1)
    ret = np.square(x)
    return ret
np1 = hf_task(np0) #printout to indicate that the calling is cached
np2 = hf_task(np0) #from cache


job_list = (joblib.delayed(math.sqrt)(x**2) for x in range(10))
x0 = joblib.Parallel(n_jobs=4)(job_list)
