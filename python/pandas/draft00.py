import numpy as np
import pandas as pd

# create
pd.Series([1, 3, 5,np.nan, 6, 8])
pd.date_range('20190217', periods=6)
pd.DataFrame(np.random.rand(6,4),
        index=pd.date_range('20190217', periods=6),
        columns=['A','B','C','D'])
pd.DataFrame({'A': 1,
        'B': pd.Timestamp('20130102'),
        'C': pd.Series(1, index=[0,1,2,3], dtype='float32'),
        'D': np.array([3,3,3,3]),
        'E': pd.Categorical(['test','train','test','train']),
        'F': 'foo'})


# property
d1 = pd.DataFrame({'A': 1,
        'B': pd.Timestamp('20130102'),
        'C': pd.Series(1, index=[0,1,2,3], dtype='float32'),
        'D': np.array([3,3,3,3]),
        'E': pd.Categorical(['test','train','test','train']),
        'F': 'foo'})
d1.dtypes
d1.head()
d1.tail(3)
d1.index
d1.columns
# d1.to_numpy() #not recommand
d1.describe()
# d1.T
# d1.transpose()
d1.sort_index(axis=1, ascending=False)
d1.sort_values('B')

s1 = pd.Series([1, 3, 5,np.nan, 6, 8])
s1.dtypes
s1.head()
s1.tail(3)
s1.index
s1.describe()
