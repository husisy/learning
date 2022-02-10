import functools

@functools.lru_cache(maxsize=128)
def hf0(x:str):
    print(x)
    ret = x + '233'
    return ret
_ = hf0('a') #print('a')
_ = hf0('b') #print('b')
_ = hf0('a') #not print()


hf1 = functools.partial(int, base=2)

functools.reduce(lambda x,y: x+y, range(11))

def str2int(s):
    tmp1 = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    hf1 = lambda x: tmp1[x]
    return functools.reduce(lambda x,y: x * 10 + y, map(hf1, s))
