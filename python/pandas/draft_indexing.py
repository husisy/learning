import numpy as np
import pandas as pd


sr1 = pd.Series([1,2,3], index=['a','b','c'])
ind1 = pd.date_range('1/1/2000', periods=8)
df1 = pd.DataFrame(np.random.randn(8,4), index=ind1, columns=['A','B','C','D'])

sr1['a']
sr1.a #np.float64, not recommand
sr1[['a','b']] #Series

df1['A'] #series
df1.A #Series, not recommand
df1[['A','B']] #DataFrame

df1.iloc[ind1[5]]

# assign
df1['E'] = [233]*len(df1.index)
df1.iloc[0] = {'A':2,'B':3,'C':3} #np.nan for absent key
df1[['B','A']] = df1[['A','B']] #in-place
df1.loc[:, ['B','A']] = df1[['A','B']] #out-place
df1.loc[:, ['B','A']] = df1[['A','B']].to_numpy() #in-place



ind1 = pd.date_range('20130101', periods=6)
d1 = pd.DataFrame(np.random.rand(6,4), index=ind1, columns=['A','B','C','D'])

d1['A']
d1.A #not recommand
d1[0:3]
d1['20130102':'20130104']
d1.loc[ind1[0]]
d1.loc[:, ['A','B']]
d1.loc['20130102':'20130104', ['A','B']]
d1.loc['20130102', ['A','B']]
