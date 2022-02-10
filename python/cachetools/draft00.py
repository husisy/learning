import time
import cachetools

@cachetools.cached(cache=cachetools.TTLCache(maxsize=512, ttl=0.3))
def hf0(key):
    print('truly call hf0()')
    return '233' + key

print('first call: ', end='')
hf0('a')
print('second call')
hf0('a')
print('sleep 0.4 second')
time.sleep(0.4)
print('third call: ', end='')
hf0('a')
