import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

np1 = np.random.rand(100,4)
hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))

trans1 = FunctionTransformer(lambda x: x[:,::-1])
trans2 = FunctionTransformer(lambda x: np.stack([x[:,0]+x[:,2],x[:,1]+x[:,3]], axis=1))
pipeline = make_pipeline(trans1, trans2)

np2 = pipeline.transform(np1)
np2_ = np.stack([np1[:,1]+np1[:,3],np1[:,0]+np1[:,2]], axis=1)
print('relative error: ', hfe(np2,np2_))
