import json

json.dumps(['foo',{'bar':('baz',None,1)}])

# dict {}
# list []
# str ""
# int/float 1234.56
# True/False true/flase
# None null
tmp1 = {y:x for x,y in enumerate('abcde')} #if x:y, results will be a little different
tmp2 = json.loads(json.dumps(tmp1))
with open(hf_tbd('tbd01.json'), 'w') as fid:
    json.dump(tmp1, fid)
with open(hf_tbd('tbd01.json'), 'r') as fid:
    tmp3 = json.load(fid)

class Class00(object):
    def __init__(self, x):
        self.x = x
x1 = Class00(233)
tmp1 = json.dumps(x1, default=lambda x:{'x':x.x})
x2 = json.loads(tmp1, object_hook=lambda x:Class00(x['x']))
tmp1 = json.dumps(x1, default=lambda x:x.__dict__)
x3 = json.loads(tmp1, object_hook=lambda x:Class00(x['x']))
