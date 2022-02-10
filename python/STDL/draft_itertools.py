import itertools


ind1 = 0
for x in itertools.count(1,2):
    print(x)
    ind1 += 1
    if ind1>10: break #otherwise, itertools.count() will never end


x1 = itertools.count(0, 2)
list(itertools.takewhile(lambda x:x<=10, x1))


ind1 = 0
for x in itertools.cycle('ABC'):
    print(x)
    ind1 += 1
    if ind1>10: break #otherwise, itertools.count() will never end


for x in itertools.repeat('ABC', 3):
    print(x)


for x in itertools.chain('ABC', 'abcd'):
    print(x)


for key,group in itertools.groupby('ABBCCCDDDD'):
    print(key, list(group))


for key,group in itertools.groupby('ABbCcCDdDd', lambda x: x.upper()):
    print(key, list(group))
